/* memcpy with SSSE3
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
   Contributed by Intel Corporation.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#if IS_IN (libc) \
    && (defined SHARED \
	|| defined USE_AS_MEMMOVE \
	|| !defined USE_MULTIARCH)

# include <sysdep.h>
# include "asm-syntax.h"

# ifndef MEMCPY
#  define MEMCPY		__memcpy_ssse3
#  define MEMCPY_CHK	__memcpy_chk_ssse3
# endif

# ifdef USE_AS_BCOPY
#  define SRC		PARMS
#  define DEST		SRC+4
#  define LEN		DEST+4
# else
#  define DEST		PARMS
#  define SRC		DEST+4
#  define LEN		SRC+4
# endif

# define CFI_PUSH(REG)		\
  cfi_adjust_cfa_offset (4);		\
  cfi_rel_offset (REG, 0)

# define CFI_POP(REG)		\
  cfi_adjust_cfa_offset (-4);		\
  cfi_restore (REG)

# define PUSH(REG)	pushl REG; CFI_PUSH (REG)
# define POP(REG)	popl REG; CFI_POP (REG)

# ifdef PIC
#  define PARMS		8		/* Preserve EBX.  */
#  define ENTRANCE	PUSH (%ebx);
#  define RETURN_END	POP (%ebx); ret
#  define RETURN		RETURN_END; CFI_PUSH (%ebx)
#  define JMPTBL(I, B)	I - B

/* Load an entry in a jump table into EBX and branch to it.  TABLE is a
	jump table with relative offsets.  INDEX is a register contains the
	index into the jump table.   SCALE is the scale of INDEX. */

#  define BRANCH_TO_JMPTBL_ENTRY(TABLE, INDEX, SCALE)		\
    /* We first load PC into EBX.  */		\
	SETUP_PIC_REG(bx);		\
    /* Get the address of the jump table.  */		\
	addl	$(TABLE - .), %ebx;		\
    /* Get the entry and convert the relative offset to the		\
	absolute	address.  */		\
	addl	(%ebx, INDEX, SCALE), %ebx;		\
    /* We loaded the jump table.  Go.  */		\
	_CET_NOTRACK jmp *%ebx
# else

#  define PARMS		4
#  define ENTRANCE
#  define RETURN_END	ret
#  define RETURN		RETURN_END
#  define JMPTBL(I, B)	I

/* Branch to an entry in a jump table.  TABLE is a jump table with
	absolute offsets.  INDEX is a register contains the index into the
	jump table.  SCALE is the scale of INDEX. */

#  define BRANCH_TO_JMPTBL_ENTRY(TABLE, INDEX, SCALE)		\
	_CET_NOTRACK jmp *TABLE(, INDEX, SCALE)
# endif

	.section .text.ssse3,"ax",@progbits
# if !defined USE_AS_BCOPY && defined SHARED
ENTRY (MEMCPY_CHK)
	movl	12(%esp), %eax
	cmpl	%eax, 16(%esp)
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (MEMCPY_CHK)
# endif
ENTRY (MEMCPY)
	ENTRANCE
	movl	LEN(%esp), %ecx
	movl	SRC(%esp), %eax
	movl	DEST(%esp), %edx

# ifdef USE_AS_MEMMOVE
	cmp	%eax, %edx
	jb	L(copy_forward)
	je	L(fwd_write_0bytes)
	cmp	$32, %ecx
	jae	L(memmove_bwd)
	jmp	L(bk_write_less32bytes_2)

	.p2align 4
L(memmove_bwd):
	add	%ecx, %eax
	cmp	%eax, %edx
	movl	SRC(%esp), %eax
	jb	L(copy_backward)

L(copy_forward):
# endif
	cmp	$48, %ecx
	jae	L(48bytesormore)

L(fwd_write_less32bytes):
# ifndef USE_AS_MEMMOVE
	cmp	%dl, %al
	jb	L(bk_write)
# endif
	add	%ecx, %edx
	add	%ecx, %eax
	BRANCH_TO_JMPTBL_ENTRY (L(table_48bytes_fwd), %ecx, 4)
# ifndef USE_AS_MEMMOVE
	.p2align 4
L(bk_write):
	BRANCH_TO_JMPTBL_ENTRY (L(table_48_bytes_bwd), %ecx, 4)
# endif

	.p2align 4
L(48bytesormore):
# ifndef USE_AS_MEMMOVE
	movlpd	(%eax), %xmm0
	movlpd	8(%eax), %xmm1
	movlpd	%xmm0, (%edx)
	movlpd	%xmm1, 8(%edx)
# else
	movdqu	(%eax), %xmm0
# endif
	PUSH (%edi)
	movl	%edx, %edi
	and	$-16, %edx
	add	$16, %edx
	sub	%edx, %edi
	add	%edi, %ecx
	sub	%edi, %eax

# ifdef SHARED_CACHE_SIZE_HALF
	cmp	$SHARED_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_shared_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_shared_cache_size_half, %ecx
#  endif
# endif

	mov	%eax, %edi
	jae	L(large_page)
	and	$0xf, %edi
	jz	L(shl_0)
	BRANCH_TO_JMPTBL_ENTRY (L(shl_table), %edi, 4)

	.p2align 4
L(shl_0):
# ifdef USE_AS_MEMMOVE
	movl	DEST+4(%esp), %edi
	movdqu	%xmm0, (%edi)
# endif
	xor	%edi, %edi
	cmp	$127, %ecx
	ja	L(shl_0_gobble)
	lea	-32(%ecx), %ecx

	.p2align 4
L(shl_0_loop):
	movdqa	(%eax, %edi), %xmm0
	movdqa	16(%eax, %edi), %xmm1
	sub	$32, %ecx
	movdqa	%xmm0, (%edx, %edi)
	movdqa	%xmm1, 16(%edx, %edi)
	lea	32(%edi), %edi
	jb	L(shl_0_end)

	movdqa	(%eax, %edi), %xmm0
	movdqa	16(%eax, %edi), %xmm1
	sub	$32, %ecx
	movdqa	%xmm0, (%edx, %edi)
	movdqa	%xmm1, 16(%edx, %edi)
	lea	32(%edi), %edi
	jb	L(shl_0_end)

	movdqa	(%eax, %edi), %xmm0
	movdqa	16(%eax, %edi), %xmm1
	sub	$32, %ecx
	movdqa	%xmm0, (%edx, %edi)
	movdqa	%xmm1, 16(%edx, %edi)
	lea	32(%edi), %edi
	jb	L(shl_0_end)

	movdqa	(%eax, %edi), %xmm0
	movdqa	16(%eax, %edi), %xmm1
	sub	$32, %ecx
	movdqa	%xmm0, (%edx, %edi)
	movdqa	%xmm1, 16(%edx, %edi)
	lea	32(%edi), %edi

L(shl_0_end):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	add	%edi, %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY (L(table_48bytes_fwd_align), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_0_gobble):
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	POP	(%edi)
	lea	-128(%ecx), %ecx
	jae	L(shl_0_gobble_mem_loop)

	.p2align 4
L(shl_0_gobble_cache_loop):
	movdqa	(%eax), %xmm0
	movdqa	0x10(%eax), %xmm1
	movdqa	0x20(%eax), %xmm2
	movdqa	0x30(%eax), %xmm3
	movdqa	0x40(%eax), %xmm4
	movdqa	0x50(%eax), %xmm5
	movdqa	0x60(%eax), %xmm6
	movdqa	0x70(%eax), %xmm7
	lea	0x80(%eax), %eax
	sub	$128, %ecx
	movdqa	%xmm0, (%edx)
	movdqa	%xmm1, 0x10(%edx)
	movdqa	%xmm2, 0x20(%edx)
	movdqa	%xmm3, 0x30(%edx)
	movdqa	%xmm4, 0x40(%edx)
	movdqa	%xmm5, 0x50(%edx)
	movdqa	%xmm6, 0x60(%edx)
	movdqa	%xmm7, 0x70(%edx)
	lea	0x80(%edx), %edx

	jae	L(shl_0_gobble_cache_loop)
	cmp	$-0x40, %ecx
	lea	0x80(%ecx), %ecx
	jl	L(shl_0_cache_less_64bytes)

	movdqa	(%eax), %xmm0
	sub	$0x40, %ecx
	movdqa	0x10(%eax), %xmm1
	movdqa	%xmm0, (%edx)
	movdqa	%xmm1, 0x10(%edx)
	movdqa	0x20(%eax), %xmm0
	movdqa	0x30(%eax), %xmm1
	add	$0x40, %eax
	movdqa	%xmm0, 0x20(%edx)
	movdqa	%xmm1, 0x30(%edx)
	add	$0x40, %edx

L(shl_0_cache_less_64bytes):
	cmp	$0x20, %ecx
	jb	L(shl_0_cache_less_32bytes)
	movdqa	(%eax), %xmm0
	sub	$0x20, %ecx
	movdqa	0x10(%eax), %xmm1
	add	$0x20, %eax
	movdqa	%xmm0, (%edx)
	movdqa	%xmm1, 0x10(%edx)
	add	$0x20, %edx

L(shl_0_cache_less_32bytes):
	cmp	$0x10, %ecx
	jb	L(shl_0_cache_less_16bytes)
	sub	$0x10, %ecx
	movdqa	(%eax), %xmm0
	add	$0x10, %eax
	movdqa	%xmm0, (%edx)
	add	$0x10, %edx

L(shl_0_cache_less_16bytes):
	add	%ecx, %edx
	add	%ecx, %eax
	BRANCH_TO_JMPTBL_ENTRY (L(table_48bytes_fwd), %ecx, 4)

	.p2align 4
L(shl_0_gobble_mem_loop):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x280(%eax)
	prefetcht0 0x1c0(%edx)

	movdqa	(%eax), %xmm0
	movdqa	0x10(%eax), %xmm1
	movdqa	0x20(%eax), %xmm2
	movdqa	0x30(%eax), %xmm3
	movdqa	0x40(%eax), %xmm4
	movdqa	0x50(%eax), %xmm5
	movdqa	0x60(%eax), %xmm6
	movdqa	0x70(%eax), %xmm7
	lea	0x80(%eax), %eax
	sub	$0x80, %ecx
	movdqa	%xmm0, (%edx)
	movdqa	%xmm1, 0x10(%edx)
	movdqa	%xmm2, 0x20(%edx)
	movdqa	%xmm3, 0x30(%edx)
	movdqa	%xmm4, 0x40(%edx)
	movdqa	%xmm5, 0x50(%edx)
	movdqa	%xmm6, 0x60(%edx)
	movdqa	%xmm7, 0x70(%edx)
	lea	0x80(%edx), %edx

	jae	L(shl_0_gobble_mem_loop)
	cmp	$-0x40, %ecx
	lea	0x80(%ecx), %ecx
	jl	L(shl_0_mem_less_64bytes)

	movdqa	(%eax), %xmm0
	sub	$0x40, %ecx
	movdqa	0x10(%eax), %xmm1

	movdqa	%xmm0, (%edx)
	movdqa	%xmm1, 0x10(%edx)

	movdqa	0x20(%eax), %xmm0
	movdqa	0x30(%eax), %xmm1
	add	$0x40, %eax

	movdqa	%xmm0, 0x20(%edx)
	movdqa	%xmm1, 0x30(%edx)
	add	$0x40, %edx

L(shl_0_mem_less_64bytes):
	cmp	$0x20, %ecx
	jb	L(shl_0_mem_less_32bytes)
	movdqa	(%eax), %xmm0
	sub	$0x20, %ecx
	movdqa	0x10(%eax), %xmm1
	add	$0x20, %eax
	movdqa	%xmm0, (%edx)
	movdqa	%xmm1, 0x10(%edx)
	add	$0x20, %edx

L(shl_0_mem_less_32bytes):
	cmp	$0x10, %ecx
	jb	L(shl_0_mem_less_16bytes)
	sub	$0x10, %ecx
	movdqa	(%eax), %xmm0
	add	$0x10, %eax
	movdqa	%xmm0, (%edx)
	add	$0x10, %edx

L(shl_0_mem_less_16bytes):
	add	%ecx, %edx
	add	%ecx, %eax
	BRANCH_TO_JMPTBL_ENTRY (L(table_48bytes_fwd_align), %ecx, 4)

	.p2align 4
L(shl_1):
# ifndef USE_AS_MEMMOVE
	movaps	-1(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-1(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_1_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl1LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	15(%eax), %xmm2
	movaps	31(%eax), %xmm3
	movaps	47(%eax), %xmm4
	movaps	63(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$1, %xmm4, %xmm5
	palignr	$1, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$1, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$1, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl1LoopStart)

L(Shl1LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	15(%eax), %xmm2
	movaps	31(%eax), %xmm3
	palignr	$1, %xmm2, %xmm3
	palignr	$1, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_1_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-1(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_1_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$1, %xmm2, %xmm3
	palignr	$1, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jb	L(sh_1_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$1, %xmm2, %xmm3
	palignr	$1, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jae	L(sh_1_no_prefetch_loop)

L(sh_1_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	1(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_2):
# ifndef USE_AS_MEMMOVE
	movaps	-2(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-2(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_2_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl2LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	14(%eax), %xmm2
	movaps	30(%eax), %xmm3
	movaps	46(%eax), %xmm4
	movaps	62(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$2, %xmm4, %xmm5
	palignr	$2, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$2, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$2, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl2LoopStart)

L(Shl2LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	14(%eax), %xmm2
	movaps	30(%eax), %xmm3
	palignr	$2, %xmm2, %xmm3
	palignr	$2, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_2_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-2(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_2_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$2, %xmm2, %xmm3
	palignr	$2, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jb	L(sh_2_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$2, %xmm2, %xmm3
	palignr	$2, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jae	L(sh_2_no_prefetch_loop)

L(sh_2_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	2(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_3):
# ifndef USE_AS_MEMMOVE
	movaps	-3(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-3(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_3_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl3LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	13(%eax), %xmm2
	movaps	29(%eax), %xmm3
	movaps	45(%eax), %xmm4
	movaps	61(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$3, %xmm4, %xmm5
	palignr	$3, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$3, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$3, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl3LoopStart)

L(Shl3LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	13(%eax), %xmm2
	movaps	29(%eax), %xmm3
	palignr	$3, %xmm2, %xmm3
	palignr	$3, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_3_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-3(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_3_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$3, %xmm2, %xmm3
	palignr	$3, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)

	jb	L(sh_3_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$3, %xmm2, %xmm3
	palignr	$3, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)

	jae	L(sh_3_no_prefetch_loop)

L(sh_3_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	3(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_4):
# ifndef USE_AS_MEMMOVE
	movaps	-4(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-4(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_4_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl4LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	12(%eax), %xmm2
	movaps	28(%eax), %xmm3
	movaps	44(%eax), %xmm4
	movaps	60(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$4, %xmm4, %xmm5
	palignr	$4, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$4, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$4, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl4LoopStart)

L(Shl4LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	12(%eax), %xmm2
	movaps	28(%eax), %xmm3
	palignr	$4, %xmm2, %xmm3
	palignr	$4, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_4_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-4(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_4_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$4, %xmm2, %xmm3
	palignr	$4, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)

	jb	L(sh_4_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$4, %xmm2, %xmm3
	palignr	$4, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)

	jae	L(sh_4_no_prefetch_loop)

L(sh_4_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	4(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_5):
# ifndef USE_AS_MEMMOVE
	movaps	-5(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-5(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_5_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl5LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	11(%eax), %xmm2
	movaps	27(%eax), %xmm3
	movaps	43(%eax), %xmm4
	movaps	59(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$5, %xmm4, %xmm5
	palignr	$5, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$5, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$5, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl5LoopStart)

L(Shl5LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	11(%eax), %xmm2
	movaps	27(%eax), %xmm3
	palignr	$5, %xmm2, %xmm3
	palignr	$5, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_5_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-5(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_5_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$5, %xmm2, %xmm3
	palignr	$5, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)

	jb	L(sh_5_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$5, %xmm2, %xmm3
	palignr	$5, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)

	jae	L(sh_5_no_prefetch_loop)

L(sh_5_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	5(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_6):
# ifndef USE_AS_MEMMOVE
	movaps	-6(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-6(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_6_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl6LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	10(%eax), %xmm2
	movaps	26(%eax), %xmm3
	movaps	42(%eax), %xmm4
	movaps	58(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$6, %xmm4, %xmm5
	palignr	$6, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$6, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$6, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl6LoopStart)

L(Shl6LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	10(%eax), %xmm2
	movaps	26(%eax), %xmm3
	palignr	$6, %xmm2, %xmm3
	palignr	$6, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_6_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-6(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_6_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$6, %xmm2, %xmm3
	palignr	$6, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)

	jb	L(sh_6_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$6, %xmm2, %xmm3
	palignr	$6, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)

	jae	L(sh_6_no_prefetch_loop)

L(sh_6_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	6(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_7):
# ifndef USE_AS_MEMMOVE
	movaps	-7(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-7(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_7_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl7LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	9(%eax), %xmm2
	movaps	25(%eax), %xmm3
	movaps	41(%eax), %xmm4
	movaps	57(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$7, %xmm4, %xmm5
	palignr	$7, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$7, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$7, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl7LoopStart)

L(Shl7LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	9(%eax), %xmm2
	movaps	25(%eax), %xmm3
	palignr	$7, %xmm2, %xmm3
	palignr	$7, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_7_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-7(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_7_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$7, %xmm2, %xmm3
	palignr	$7, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jb	L(sh_7_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$7, %xmm2, %xmm3
	palignr	$7, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jae	L(sh_7_no_prefetch_loop)

L(sh_7_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	7(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_8):
# ifndef USE_AS_MEMMOVE
	movaps	-8(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-8(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_8_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl8LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	8(%eax), %xmm2
	movaps	24(%eax), %xmm3
	movaps	40(%eax), %xmm4
	movaps	56(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$8, %xmm4, %xmm5
	palignr	$8, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$8, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$8, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl8LoopStart)

L(LoopLeave8):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	8(%eax), %xmm2
	movaps	24(%eax), %xmm3
	palignr	$8, %xmm2, %xmm3
	palignr	$8, %xmm1, %xmm2
	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_8_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-8(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_8_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$8, %xmm2, %xmm3
	palignr	$8, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jb	L(sh_8_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$8, %xmm2, %xmm3
	palignr	$8, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jae	L(sh_8_no_prefetch_loop)

L(sh_8_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	8(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_9):
# ifndef USE_AS_MEMMOVE
	movaps	-9(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-9(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_9_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl9LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	7(%eax), %xmm2
	movaps	23(%eax), %xmm3
	movaps	39(%eax), %xmm4
	movaps	55(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$9, %xmm4, %xmm5
	palignr	$9, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$9, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$9, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl9LoopStart)

L(Shl9LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	7(%eax), %xmm2
	movaps	23(%eax), %xmm3
	palignr	$9, %xmm2, %xmm3
	palignr	$9, %xmm1, %xmm2

	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_9_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-9(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_9_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$9, %xmm2, %xmm3
	palignr	$9, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jb	L(sh_9_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$9, %xmm2, %xmm3
	palignr	$9, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jae	L(sh_9_no_prefetch_loop)

L(sh_9_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	9(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_10):
# ifndef USE_AS_MEMMOVE
	movaps	-10(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-10(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_10_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl10LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	6(%eax), %xmm2
	movaps	22(%eax), %xmm3
	movaps	38(%eax), %xmm4
	movaps	54(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$10, %xmm4, %xmm5
	palignr	$10, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$10, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$10, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl10LoopStart)

L(Shl10LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	6(%eax), %xmm2
	movaps	22(%eax), %xmm3
	palignr	$10, %xmm2, %xmm3
	palignr	$10, %xmm1, %xmm2

	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_10_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-10(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_10_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$10, %xmm2, %xmm3
	palignr	$10, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jb	L(sh_10_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$10, %xmm2, %xmm3
	palignr	$10, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jae	L(sh_10_no_prefetch_loop)

L(sh_10_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	10(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_11):
# ifndef USE_AS_MEMMOVE
	movaps	-11(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-11(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_11_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl11LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	5(%eax), %xmm2
	movaps	21(%eax), %xmm3
	movaps	37(%eax), %xmm4
	movaps	53(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$11, %xmm4, %xmm5
	palignr	$11, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$11, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$11, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl11LoopStart)

L(Shl11LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	5(%eax), %xmm2
	movaps	21(%eax), %xmm3
	palignr	$11, %xmm2, %xmm3
	palignr	$11, %xmm1, %xmm2

	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_11_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-11(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_11_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$11, %xmm2, %xmm3
	palignr	$11, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jb	L(sh_11_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$11, %xmm2, %xmm3
	palignr	$11, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jae	L(sh_11_no_prefetch_loop)

L(sh_11_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	11(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_12):
# ifndef USE_AS_MEMMOVE
	movaps	-12(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-12(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_12_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl12LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	4(%eax), %xmm2
	movaps	20(%eax), %xmm3
	movaps	36(%eax), %xmm4
	movaps	52(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$12, %xmm4, %xmm5
	palignr	$12, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$12, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$12, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl12LoopStart)

L(Shl12LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	4(%eax), %xmm2
	movaps	20(%eax), %xmm3
	palignr	$12, %xmm2, %xmm3
	palignr	$12, %xmm1, %xmm2

	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_12_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-12(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_12_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$12, %xmm2, %xmm3
	palignr	$12, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jb	L(sh_12_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$12, %xmm2, %xmm3
	palignr	$12, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jae	L(sh_12_no_prefetch_loop)

L(sh_12_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	12(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_13):
# ifndef USE_AS_MEMMOVE
	movaps	-13(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-13(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_13_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl13LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	3(%eax), %xmm2
	movaps	19(%eax), %xmm3
	movaps	35(%eax), %xmm4
	movaps	51(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$13, %xmm4, %xmm5
	palignr	$13, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$13, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$13, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl13LoopStart)

L(Shl13LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	3(%eax), %xmm2
	movaps	19(%eax), %xmm3
	palignr	$13, %xmm2, %xmm3
	palignr	$13, %xmm1, %xmm2

	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_13_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-13(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_13_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$13, %xmm2, %xmm3
	palignr	$13, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jb	L(sh_13_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$13, %xmm2, %xmm3
	palignr	$13, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jae	L(sh_13_no_prefetch_loop)

L(sh_13_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	13(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_14):
# ifndef USE_AS_MEMMOVE
	movaps	-14(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-14(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_14_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl14LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	2(%eax), %xmm2
	movaps	18(%eax), %xmm3
	movaps	34(%eax), %xmm4
	movaps	50(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$14, %xmm4, %xmm5
	palignr	$14, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$14, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$14, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl14LoopStart)

L(Shl14LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	2(%eax), %xmm2
	movaps	18(%eax), %xmm3
	palignr	$14, %xmm2, %xmm3
	palignr	$14, %xmm1, %xmm2

	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_14_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-14(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_14_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$14, %xmm2, %xmm3
	palignr	$14, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jb	L(sh_14_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$14, %xmm2, %xmm3
	palignr	$14, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jae	L(sh_14_no_prefetch_loop)

L(sh_14_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	14(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_15):
# ifndef USE_AS_MEMMOVE
	movaps	-15(%eax), %xmm1
# else
	movl	DEST+4(%esp), %edi
	movaps	-15(%eax), %xmm1
	movdqu	%xmm0, (%edi)
# endif
# ifdef DATA_CACHE_SIZE_HALF
	cmp	$DATA_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_data_cache_size_half, %ecx
#  endif
# endif
	jb L(sh_15_no_prefetch)

	lea	-64(%ecx), %ecx

	.p2align 4
L(Shl15LoopStart):
	prefetcht0 0x1c0(%eax)
	prefetcht0 0x1c0(%edx)
	movaps	1(%eax), %xmm2
	movaps	17(%eax), %xmm3
	movaps	33(%eax), %xmm4
	movaps	49(%eax), %xmm5
	movaps	%xmm5, %xmm7
	palignr	$15, %xmm4, %xmm5
	palignr	$15, %xmm3, %xmm4
	movaps	%xmm5, 48(%edx)
	palignr	$15, %xmm2, %xmm3
	lea	64(%eax), %eax
	palignr	$15, %xmm1, %xmm2
	movaps	%xmm4, 32(%edx)
	movaps	%xmm3, 16(%edx)
	movaps	%xmm7, %xmm1
	movaps	%xmm2, (%edx)
	lea	64(%edx), %edx
	sub	$64, %ecx
	ja	L(Shl15LoopStart)

L(Shl15LoopLeave):
	add	$32, %ecx
	jle	L(shl_end_0)

	movaps	1(%eax), %xmm2
	movaps	17(%eax), %xmm3
	palignr	$15, %xmm2, %xmm3
	palignr	$15, %xmm1, %xmm2

	movaps	%xmm2, (%edx)
	movaps	%xmm3, 16(%edx)
	lea	32(%edx, %ecx), %edx
	lea	32(%eax, %ecx), %eax
	POP (%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(sh_15_no_prefetch):
	lea	-32(%ecx), %ecx
	lea	-15(%eax), %eax
	xor	%edi, %edi

	.p2align 4
L(sh_15_no_prefetch_loop):
	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm4
	palignr	$15, %xmm2, %xmm3
	palignr	$15, %xmm1, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jb	L(sh_15_end_no_prefetch_loop)

	movdqa	16(%eax, %edi), %xmm2
	sub	$32, %ecx
	movdqa	32(%eax, %edi), %xmm3
	movdqa	%xmm3, %xmm1
	palignr	$15, %xmm2, %xmm3
	palignr	$15, %xmm4, %xmm2
	lea	32(%edi), %edi
	movdqa	%xmm2, -32(%edx, %edi)
	movdqa	%xmm3, -16(%edx, %edi)
	jae	L(sh_15_no_prefetch_loop)

L(sh_15_end_no_prefetch_loop):
	lea	32(%ecx), %ecx
	add	%ecx, %edi
	add	%edi, %edx
	lea	15(%edi, %eax), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(shl_end_0):
	lea	32(%ecx), %ecx
	lea	(%edx, %ecx), %edx
	lea	(%eax, %ecx), %eax
	POP	(%edi)
	BRANCH_TO_JMPTBL_ENTRY(L(table_48bytes_fwd), %ecx, 4)

	.p2align 4
L(fwd_write_44bytes):
	movq	-44(%eax), %xmm0
	movq	%xmm0, -44(%edx)
L(fwd_write_36bytes):
	movq	-36(%eax), %xmm0
	movq	%xmm0, -36(%edx)
L(fwd_write_28bytes):
	movq	-28(%eax), %xmm0
	movq	%xmm0, -28(%edx)
L(fwd_write_20bytes):
	movq	-20(%eax), %xmm0
	movq	%xmm0, -20(%edx)
L(fwd_write_12bytes):
	movq	-12(%eax), %xmm0
	movq	%xmm0, -12(%edx)
L(fwd_write_4bytes):
	movl	-4(%eax), %ecx
	movl	%ecx, -4(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_40bytes):
	movq	-40(%eax), %xmm0
	movq	%xmm0, -40(%edx)
L(fwd_write_32bytes):
	movq	-32(%eax), %xmm0
	movq	%xmm0, -32(%edx)
L(fwd_write_24bytes):
	movq	-24(%eax), %xmm0
	movq	%xmm0, -24(%edx)
L(fwd_write_16bytes):
	movq	-16(%eax), %xmm0
	movq	%xmm0, -16(%edx)
L(fwd_write_8bytes):
	movq	-8(%eax), %xmm0
	movq	%xmm0, -8(%edx)
L(fwd_write_0bytes):
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_5bytes):
	movl	-5(%eax), %ecx
	movl	-4(%eax), %eax
	movl	%ecx, -5(%edx)
	movl	%eax, -4(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_45bytes):
	movq	-45(%eax), %xmm0
	movq	%xmm0, -45(%edx)
L(fwd_write_37bytes):
	movq	-37(%eax), %xmm0
	movq	%xmm0, -37(%edx)
L(fwd_write_29bytes):
	movq	-29(%eax), %xmm0
	movq	%xmm0, -29(%edx)
L(fwd_write_21bytes):
	movq	-21(%eax), %xmm0
	movq	%xmm0, -21(%edx)
L(fwd_write_13bytes):
	movq	-13(%eax), %xmm0
	movq	%xmm0, -13(%edx)
	movl	-5(%eax), %ecx
	movl	%ecx, -5(%edx)
	movzbl	-1(%eax), %ecx
	movb	%cl, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_41bytes):
	movq	-41(%eax), %xmm0
	movq	%xmm0, -41(%edx)
L(fwd_write_33bytes):
	movq	-33(%eax), %xmm0
	movq	%xmm0, -33(%edx)
L(fwd_write_25bytes):
	movq	-25(%eax), %xmm0
	movq	%xmm0, -25(%edx)
L(fwd_write_17bytes):
	movq	-17(%eax), %xmm0
	movq	%xmm0, -17(%edx)
L(fwd_write_9bytes):
	movq	-9(%eax), %xmm0
	movq	%xmm0, -9(%edx)
L(fwd_write_1bytes):
	movzbl	-1(%eax), %ecx
	movb	%cl, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_46bytes):
	movq	-46(%eax), %xmm0
	movq	%xmm0, -46(%edx)
L(fwd_write_38bytes):
	movq	-38(%eax), %xmm0
	movq	%xmm0, -38(%edx)
L(fwd_write_30bytes):
	movq	-30(%eax), %xmm0
	movq	%xmm0, -30(%edx)
L(fwd_write_22bytes):
	movq	-22(%eax), %xmm0
	movq	%xmm0, -22(%edx)
L(fwd_write_14bytes):
	movq	-14(%eax), %xmm0
	movq	%xmm0, -14(%edx)
L(fwd_write_6bytes):
	movl	-6(%eax), %ecx
	movl	%ecx, -6(%edx)
	movzwl	-2(%eax), %ecx
	movw	%cx, -2(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_42bytes):
	movq	-42(%eax), %xmm0
	movq	%xmm0, -42(%edx)
L(fwd_write_34bytes):
	movq	-34(%eax), %xmm0
	movq	%xmm0, -34(%edx)
L(fwd_write_26bytes):
	movq	-26(%eax), %xmm0
	movq	%xmm0, -26(%edx)
L(fwd_write_18bytes):
	movq	-18(%eax), %xmm0
	movq	%xmm0, -18(%edx)
L(fwd_write_10bytes):
	movq	-10(%eax), %xmm0
	movq	%xmm0, -10(%edx)
L(fwd_write_2bytes):
	movzwl	-2(%eax), %ecx
	movw	%cx, -2(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_47bytes):
	movq	-47(%eax), %xmm0
	movq	%xmm0, -47(%edx)
L(fwd_write_39bytes):
	movq	-39(%eax), %xmm0
	movq	%xmm0, -39(%edx)
L(fwd_write_31bytes):
	movq	-31(%eax), %xmm0
	movq	%xmm0, -31(%edx)
L(fwd_write_23bytes):
	movq	-23(%eax), %xmm0
	movq	%xmm0, -23(%edx)
L(fwd_write_15bytes):
	movq	-15(%eax), %xmm0
	movq	%xmm0, -15(%edx)
L(fwd_write_7bytes):
	movl	-7(%eax), %ecx
	movl	%ecx, -7(%edx)
	movzwl	-3(%eax), %ecx
	movzbl	-1(%eax), %eax
	movw	%cx, -3(%edx)
	movb	%al, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_43bytes):
	movq	-43(%eax), %xmm0
	movq	%xmm0, -43(%edx)
L(fwd_write_35bytes):
	movq	-35(%eax), %xmm0
	movq	%xmm0, -35(%edx)
L(fwd_write_27bytes):
	movq	-27(%eax), %xmm0
	movq	%xmm0, -27(%edx)
L(fwd_write_19bytes):
	movq	-19(%eax), %xmm0
	movq	%xmm0, -19(%edx)
L(fwd_write_11bytes):
	movq	-11(%eax), %xmm0
	movq	%xmm0, -11(%edx)
L(fwd_write_3bytes):
	movzwl	-3(%eax), %ecx
	movzbl	-1(%eax), %eax
	movw	%cx, -3(%edx)
	movb	%al, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_40bytes_align):
	movdqa	-40(%eax), %xmm0
	movdqa	%xmm0, -40(%edx)
L(fwd_write_24bytes_align):
	movdqa	-24(%eax), %xmm0
	movdqa	%xmm0, -24(%edx)
L(fwd_write_8bytes_align):
	movq	-8(%eax), %xmm0
	movq	%xmm0, -8(%edx)
L(fwd_write_0bytes_align):
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_32bytes_align):
	movdqa	-32(%eax), %xmm0
	movdqa	%xmm0, -32(%edx)
L(fwd_write_16bytes_align):
	movdqa	-16(%eax), %xmm0
	movdqa	%xmm0, -16(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_5bytes_align):
	movl	-5(%eax), %ecx
	movl	-4(%eax), %eax
	movl	%ecx, -5(%edx)
	movl	%eax, -4(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_45bytes_align):
	movdqa	-45(%eax), %xmm0
	movdqa	%xmm0, -45(%edx)
L(fwd_write_29bytes_align):
	movdqa	-29(%eax), %xmm0
	movdqa	%xmm0, -29(%edx)
L(fwd_write_13bytes_align):
	movq	-13(%eax), %xmm0
	movq	%xmm0, -13(%edx)
	movl	-5(%eax), %ecx
	movl	%ecx, -5(%edx)
	movzbl	-1(%eax), %ecx
	movb	%cl, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_37bytes_align):
	movdqa	-37(%eax), %xmm0
	movdqa	%xmm0, -37(%edx)
L(fwd_write_21bytes_align):
	movdqa	-21(%eax), %xmm0
	movdqa	%xmm0, -21(%edx)
	movl	-5(%eax), %ecx
	movl	%ecx, -5(%edx)
	movzbl	-1(%eax), %ecx
	movb	%cl, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_41bytes_align):
	movdqa	-41(%eax), %xmm0
	movdqa	%xmm0, -41(%edx)
L(fwd_write_25bytes_align):
	movdqa	-25(%eax), %xmm0
	movdqa	%xmm0, -25(%edx)
L(fwd_write_9bytes_align):
	movq	-9(%eax), %xmm0
	movq	%xmm0, -9(%edx)
L(fwd_write_1bytes_align):
	movzbl	-1(%eax), %ecx
	movb	%cl, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_33bytes_align):
	movdqa	-33(%eax), %xmm0
	movdqa	%xmm0, -33(%edx)
L(fwd_write_17bytes_align):
	movdqa	-17(%eax), %xmm0
	movdqa	%xmm0, -17(%edx)
	movzbl	-1(%eax), %ecx
	movb	%cl, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_46bytes_align):
	movdqa	-46(%eax), %xmm0
	movdqa	%xmm0, -46(%edx)
L(fwd_write_30bytes_align):
	movdqa	-30(%eax), %xmm0
	movdqa	%xmm0, -30(%edx)
L(fwd_write_14bytes_align):
	movq	-14(%eax), %xmm0
	movq	%xmm0, -14(%edx)
L(fwd_write_6bytes_align):
	movl	-6(%eax), %ecx
	movl	%ecx, -6(%edx)
	movzwl	-2(%eax), %ecx
	movw	%cx, -2(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_38bytes_align):
	movdqa	-38(%eax), %xmm0
	movdqa	%xmm0, -38(%edx)
L(fwd_write_22bytes_align):
	movdqa	-22(%eax), %xmm0
	movdqa	%xmm0, -22(%edx)
	movl	-6(%eax), %ecx
	movl	%ecx, -6(%edx)
	movzwl	-2(%eax), %ecx
	movw	%cx, -2(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_42bytes_align):
	movdqa	-42(%eax), %xmm0
	movdqa	%xmm0, -42(%edx)
L(fwd_write_26bytes_align):
	movdqa	-26(%eax), %xmm0
	movdqa	%xmm0, -26(%edx)
L(fwd_write_10bytes_align):
	movq	-10(%eax), %xmm0
	movq	%xmm0, -10(%edx)
L(fwd_write_2bytes_align):
	movzwl	-2(%eax), %ecx
	movw	%cx, -2(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_34bytes_align):
	movdqa	-34(%eax), %xmm0
	movdqa	%xmm0, -34(%edx)
L(fwd_write_18bytes_align):
	movdqa	-18(%eax), %xmm0
	movdqa	%xmm0, -18(%edx)
	movzwl	-2(%eax), %ecx
	movw	%cx, -2(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_47bytes_align):
	movdqa	-47(%eax), %xmm0
	movdqa	%xmm0, -47(%edx)
L(fwd_write_31bytes_align):
	movdqa	-31(%eax), %xmm0
	movdqa	%xmm0, -31(%edx)
L(fwd_write_15bytes_align):
	movq	-15(%eax), %xmm0
	movq	%xmm0, -15(%edx)
L(fwd_write_7bytes_align):
	movl	-7(%eax), %ecx
	movl	%ecx, -7(%edx)
	movzwl	-3(%eax), %ecx
	movzbl	-1(%eax), %eax
	movw	%cx, -3(%edx)
	movb	%al, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_39bytes_align):
	movdqa	-39(%eax), %xmm0
	movdqa	%xmm0, -39(%edx)
L(fwd_write_23bytes_align):
	movdqa	-23(%eax), %xmm0
	movdqa	%xmm0, -23(%edx)
	movl	-7(%eax), %ecx
	movl	%ecx, -7(%edx)
	movzwl	-3(%eax), %ecx
	movzbl	-1(%eax), %eax
	movw	%cx, -3(%edx)
	movb	%al, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_43bytes_align):
	movdqa	-43(%eax), %xmm0
	movdqa	%xmm0, -43(%edx)
L(fwd_write_27bytes_align):
	movdqa	-27(%eax), %xmm0
	movdqa	%xmm0, -27(%edx)
L(fwd_write_11bytes_align):
	movq	-11(%eax), %xmm0
	movq	%xmm0, -11(%edx)
L(fwd_write_3bytes_align):
	movzwl	-3(%eax), %ecx
	movzbl	-1(%eax), %eax
	movw	%cx, -3(%edx)
	movb	%al, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_35bytes_align):
	movdqa	-35(%eax), %xmm0
	movdqa	%xmm0, -35(%edx)
L(fwd_write_19bytes_align):
	movdqa	-19(%eax), %xmm0
	movdqa	%xmm0, -19(%edx)
	movzwl	-3(%eax), %ecx
	movzbl	-1(%eax), %eax
	movw	%cx, -3(%edx)
	movb	%al, -1(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_44bytes_align):
	movdqa	-44(%eax), %xmm0
	movdqa	%xmm0, -44(%edx)
L(fwd_write_28bytes_align):
	movdqa	-28(%eax), %xmm0
	movdqa	%xmm0, -28(%edx)
L(fwd_write_12bytes_align):
	movq	-12(%eax), %xmm0
	movq	%xmm0, -12(%edx)
L(fwd_write_4bytes_align):
	movl	-4(%eax), %ecx
	movl	%ecx, -4(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN

	.p2align 4
L(fwd_write_36bytes_align):
	movdqa	-36(%eax), %xmm0
	movdqa	%xmm0, -36(%edx)
L(fwd_write_20bytes_align):
	movdqa	-20(%eax), %xmm0
	movdqa	%xmm0, -20(%edx)
	movl	-4(%eax), %ecx
	movl	%ecx, -4(%edx)
# ifndef USE_AS_BCOPY
#  ifdef USE_AS_MEMPCPY
	movl	%edx, %eax
#  else
	movl	DEST(%esp), %eax
#  endif
# endif
	RETURN_END

	CFI_PUSH (%edi)

	.p2align 4
L(large_page):
	movdqu	(%eax), %xmm1
# ifdef USE_AS_MEMMOVE
	movl	DEST+4(%esp), %edi
	movdqu	%xmm0, (%edi)
# endif
	lea	16(%eax), %eax
	movntdq	%xmm1, (%edx)
	lea	16(%edx), %edx
	lea	-0x90(%ecx), %ecx
	POP (%edi)

	.p2align 4
L(large_page_loop):
	movdqu	(%eax), %xmm0
	movdqu	0x10(%eax), %xmm1
	movdqu	0x20(%eax), %xmm2
	movdqu	0x30(%eax), %xmm3
	movdqu	0x40(%eax), %xmm4
	movdqu	0x50(%eax), %xmm5
	movdqu	0x60(%eax), %xmm6
	movdqu	0x70(%eax), %xmm7
	lea	0x80(%eax), %eax

	sub	$0x80, %ecx
	movntdq	%xmm0, (%edx)
	movntdq	%xmm1, 0x10(%edx)
	movntdq	%xmm2, 0x20(%edx)
	movntdq	%xmm3, 0x30(%edx)
	movntdq	%xmm4, 0x40(%edx)
	movntdq	%xmm5, 0x50(%edx)
	movntdq	%xmm6, 0x60(%edx)
	movntdq	%xmm7, 0x70(%edx)
	lea	0x80(%edx), %edx
	jae	L(large_page_loop)
	cmp	$-0x40, %ecx
	lea	0x80(%ecx), %ecx
	jl	L(large_page_less_64bytes)

	movdqu	(%eax), %xmm0
	movdqu	0x10(%eax), %xmm1
	movdqu	0x20(%eax), %xmm2
	movdqu	0x30(%eax), %xmm3
	lea	0x40(%eax), %eax

	movntdq	%xmm0, (%edx)
	movntdq	%xmm1, 0x10(%edx)
	movntdq	%xmm2, 0x20(%edx)
	movntdq	%xmm3, 0x30(%edx)
	lea	0x40(%edx), %edx
	sub	$0x40, %ecx
L(large_page_less_64bytes):
	cmp	$32, %ecx
	jb	L(large_page_less_32bytes)
	movdqu	(%eax), %xmm0
	movdqu	0x10(%eax), %xmm1
	lea	0x20(%eax), %eax
	movntdq	%xmm0, (%edx)
	movntdq	%xmm1, 0x10(%edx)
	lea	0x20(%edx), %edx
	sub	$0x20, %ecx
L(large_page_less_32bytes):
	add	%ecx, %edx
	add	%ecx, %eax
	sfence
	BRANCH_TO_JMPTBL_ENTRY (L(table_48bytes_fwd), %ecx, 4)

	.p2align 4
L(bk_write_44bytes):
	movq	36(%eax), %xmm0
	movq	%xmm0, 36(%edx)
L(bk_write_36bytes):
	movq	28(%eax), %xmm0
	movq	%xmm0, 28(%edx)
L(bk_write_28bytes):
	movq	20(%eax), %xmm0
	movq	%xmm0, 20(%edx)
L(bk_write_20bytes):
	movq	12(%eax), %xmm0
	movq	%xmm0, 12(%edx)
L(bk_write_12bytes):
	movq	4(%eax), %xmm0
	movq	%xmm0, 4(%edx)
L(bk_write_4bytes):
	movl	(%eax), %ecx
	movl	%ecx, (%edx)
L(bk_write_0bytes):
# ifndef USE_AS_BCOPY
	movl	DEST(%esp), %eax
#  ifdef USE_AS_MEMPCPY
	movl	LEN(%esp), %ecx
	add	%ecx, %eax
#  endif
# endif
	RETURN

	.p2align 4
L(bk_write_40bytes):
	movq	32(%eax), %xmm0
	movq	%xmm0, 32(%edx)
L(bk_write_32bytes):
	movq	24(%eax), %xmm0
	movq	%xmm0, 24(%edx)
L(bk_write_24bytes):
	movq	16(%eax), %xmm0
	movq	%xmm0, 16(%edx)
L(bk_write_16bytes):
	movq	8(%eax), %xmm0
	movq	%xmm0, 8(%edx)
L(bk_write_8bytes):
	movq	(%eax), %xmm0
	movq	%xmm0, (%edx)
# ifndef USE_AS_BCOPY
	movl	DEST(%esp), %eax
#  ifdef USE_AS_MEMPCPY
	movl	LEN(%esp), %ecx
	add	%ecx, %eax
#  endif
# endif
	RETURN

	.p2align 4
L(bk_write_45bytes):
	movq	37(%eax), %xmm0
	movq	%xmm0, 37(%edx)
L(bk_write_37bytes):
	movq	29(%eax), %xmm0
	movq	%xmm0, 29(%edx)
L(bk_write_29bytes):
	movq	21(%eax), %xmm0
	movq	%xmm0, 21(%edx)
L(bk_write_21bytes):
	movq	13(%eax), %xmm0
	movq	%xmm0, 13(%edx)
L(bk_write_13bytes):
	movq	5(%eax), %xmm0
	movq	%xmm0, 5(%edx)
L(bk_write_5bytes):
	movl	1(%eax), %ecx
	movl	%ecx, 1(%edx)
L(bk_write_1bytes):
	movzbl	(%eax), %ecx
	movb	%cl, (%edx)
# ifndef USE_AS_BCOPY
	movl	DEST(%esp), %eax
#  ifdef USE_AS_MEMPCPY
	movl	LEN(%esp), %ecx
	add	%ecx, %eax
#  endif
# endif
	RETURN

	.p2align 4
L(bk_write_41bytes):
	movq	33(%eax), %xmm0
	movq	%xmm0, 33(%edx)
L(bk_write_33bytes):
	movq	25(%eax), %xmm0
	movq	%xmm0, 25(%edx)
L(bk_write_25bytes):
	movq	17(%eax), %xmm0
	movq	%xmm0, 17(%edx)
L(bk_write_17bytes):
	movq	9(%eax), %xmm0
	movq	%xmm0, 9(%edx)
L(bk_write_9bytes):
	movq	1(%eax), %xmm0
	movq	%xmm0, 1(%edx)
	movzbl	(%eax), %ecx
	movb	%cl, (%edx)
# ifndef USE_AS_BCOPY
	movl	DEST(%esp), %eax
#  ifdef USE_AS_MEMPCPY
	movl	LEN(%esp), %ecx
	add	%ecx, %eax
#  endif
# endif
	RETURN

	.p2align 4
L(bk_write_46bytes):
	movq	38(%eax), %xmm0
	movq	%xmm0, 38(%edx)
L(bk_write_38bytes):
	movq	30(%eax), %xmm0
	movq	%xmm0, 30(%edx)
L(bk_write_30bytes):
	movq	22(%eax), %xmm0
	movq	%xmm0, 22(%edx)
L(bk_write_22bytes):
	movq	14(%eax), %xmm0
	movq	%xmm0, 14(%edx)
L(bk_write_14bytes):
	movq	6(%eax), %xmm0
	movq	%xmm0, 6(%edx)
L(bk_write_6bytes):
	movl	2(%eax), %ecx
	movl	%ecx, 2(%edx)
	movzwl	(%eax), %ecx
	movw	%cx, (%edx)
# ifndef USE_AS_BCOPY
	movl	DEST(%esp), %eax
#  ifdef USE_AS_MEMPCPY
	movl	LEN(%esp), %ecx
	add	%ecx, %eax
#  endif
# endif
	RETURN

	.p2align 4
L(bk_write_42bytes):
	movq	34(%eax), %xmm0
	movq	%xmm0, 34(%edx)
L(bk_write_34bytes):
	movq	26(%eax), %xmm0
	movq	%xmm0, 26(%edx)
L(bk_write_26bytes):
	movq	18(%eax), %xmm0
	movq	%xmm0, 18(%edx)
L(bk_write_18bytes):
	movq	10(%eax), %xmm0
	movq	%xmm0, 10(%edx)
L(bk_write_10bytes):
	movq	2(%eax), %xmm0
	movq	%xmm0, 2(%edx)
L(bk_write_2bytes):
	movzwl	(%eax), %ecx
	movw	%cx, (%edx)
# ifndef USE_AS_BCOPY
	movl	DEST(%esp), %eax
#  ifdef USE_AS_MEMPCPY
	movl	LEN(%esp), %ecx
	add	%ecx, %eax
#  endif
# endif
	RETURN

	.p2align 4
L(bk_write_47bytes):
	movq	39(%eax), %xmm0
	movq	%xmm0, 39(%edx)
L(bk_write_39bytes):
	movq	31(%eax), %xmm0
	movq	%xmm0, 31(%edx)
L(bk_write_31bytes):
	movq	23(%eax), %xmm0
	movq	%xmm0, 23(%edx)
L(bk_write_23bytes):
	movq	15(%eax), %xmm0
	movq	%xmm0, 15(%edx)
L(bk_write_15bytes):
	movq	7(%eax), %xmm0
	movq	%xmm0, 7(%edx)
L(bk_write_7bytes):
	movl	3(%eax), %ecx
	movl	%ecx, 3(%edx)
	movzwl	1(%eax), %ecx
	movw	%cx, 1(%edx)
	movzbl	(%eax), %eax
	movb	%al, (%edx)
# ifndef USE_AS_BCOPY
	movl	DEST(%esp), %eax
#  ifdef USE_AS_MEMPCPY
	movl	LEN(%esp), %ecx
	add	%ecx, %eax
#  endif
# endif
	RETURN

	.p2align 4
L(bk_write_43bytes):
	movq	35(%eax), %xmm0
	movq	%xmm0, 35(%edx)
L(bk_write_35bytes):
	movq	27(%eax), %xmm0
	movq	%xmm0, 27(%edx)
L(bk_write_27bytes):
	movq	19(%eax), %xmm0
	movq	%xmm0, 19(%edx)
L(bk_write_19bytes):
	movq	11(%eax), %xmm0
	movq	%xmm0, 11(%edx)
L(bk_write_11bytes):
	movq	3(%eax), %xmm0
	movq	%xmm0, 3(%edx)
L(bk_write_3bytes):
	movzwl	1(%eax), %ecx
	movw	%cx, 1(%edx)
	movzbl	(%eax), %eax
	movb	%al, (%edx)
# ifndef USE_AS_BCOPY
	movl	DEST(%esp), %eax
#  ifdef USE_AS_MEMPCPY
	movl	LEN(%esp), %ecx
	add	%ecx, %eax
#  endif
# endif
	RETURN_END


	.pushsection .rodata.ssse3,"a",@progbits
	.p2align 2
L(table_48bytes_fwd):
	.int	JMPTBL (L(fwd_write_0bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_1bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_2bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_3bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_4bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_5bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_6bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_7bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_8bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_9bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_10bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_11bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_12bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_13bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_14bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_15bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_16bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_17bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_18bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_19bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_20bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_21bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_22bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_23bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_24bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_25bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_26bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_27bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_28bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_29bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_30bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_31bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_32bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_33bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_34bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_35bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_36bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_37bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_38bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_39bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_40bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_41bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_42bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_43bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_44bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_45bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_46bytes), L(table_48bytes_fwd))
	.int	JMPTBL (L(fwd_write_47bytes), L(table_48bytes_fwd))

	.p2align 2
L(table_48bytes_fwd_align):
	.int	JMPTBL (L(fwd_write_0bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_1bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_2bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_3bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_4bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_5bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_6bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_7bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_8bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_9bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_10bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_11bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_12bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_13bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_14bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_15bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_16bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_17bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_18bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_19bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_20bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_21bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_22bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_23bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_24bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_25bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_26bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_27bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_28bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_29bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_30bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_31bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_32bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_33bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_34bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_35bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_36bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_37bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_38bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_39bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_40bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_41bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_42bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_43bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_44bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_45bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_46bytes_align), L(table_48bytes_fwd_align))
	.int	JMPTBL (L(fwd_write_47bytes_align), L(table_48bytes_fwd_align))

	.p2align 2
L(shl_table):
	.int	JMPTBL (L(shl_0), L(shl_table))
	.int	JMPTBL (L(shl_1), L(shl_table))
	.int	JMPTBL (L(shl_2), L(shl_table))
	.int	JMPTBL (L(shl_3), L(shl_table))
	.int	JMPTBL (L(shl_4), L(shl_table))
	.int	JMPTBL (L(shl_5), L(shl_table))
	.int	JMPTBL (L(shl_6), L(shl_table))
	.int	JMPTBL (L(shl_7), L(shl_table))
	.int	JMPTBL (L(shl_8), L(shl_table))
	.int	JMPTBL (L(shl_9), L(shl_table))
	.int	JMPTBL (L(shl_10), L(shl_table))
	.int	JMPTBL (L(shl_11), L(shl_table))
	.int	JMPTBL (L(shl_12), L(shl_table))
	.int	JMPTBL (L(shl_13), L(shl_table))
	.int	JMPTBL (L(shl_14), L(shl_table))
	.int	JMPTBL (L(shl_15), L(shl_table))

	.p2align 2
L(table_48_bytes_bwd):
	.int	JMPTBL (L(bk_write_0bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_1bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_2bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_3bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_4bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_5bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_6bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_7bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_8bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_9bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_10bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_11bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_12bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_13bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_14bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_15bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_16bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_17bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_18bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_19bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_20bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_21bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_22bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_23bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_24bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_25bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_26bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_27bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_28bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_29bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_30bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_31bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_32bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_33bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_34bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_35bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_36bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_37bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_38bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_39bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_40bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_41bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_42bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_43bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_44bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_45bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_46bytes), L(table_48_bytes_bwd))
	.int	JMPTBL (L(bk_write_47bytes), L(table_48_bytes_bwd))

	.popsection

# ifdef USE_AS_MEMMOVE
	.p2align 4
L(copy_backward):
	PUSH (%edi)
	movl	%eax, %edi
	lea	(%ecx,%edx,1),%edx
	lea	(%ecx,%edi,1),%edi
	testl	$0x3, %edx
	jnz	L(bk_align)

L(bk_aligned_4):
	cmp	$64, %ecx
	jae	L(bk_write_more64bytes)

L(bk_write_64bytesless):
	cmp	$32, %ecx
	jb	L(bk_write_less32bytes)

L(bk_write_more32bytes):
	/* Copy 32 bytes at a time.  */
	sub	$32, %ecx
	movq	-8(%edi), %xmm0
	movq	%xmm0, -8(%edx)
	movq	-16(%edi), %xmm0
	movq	%xmm0, -16(%edx)
	movq	-24(%edi), %xmm0
	movq	%xmm0, -24(%edx)
	movq	-32(%edi), %xmm0
	movq	%xmm0, -32(%edx)
	sub	$32, %edx
	sub	$32, %edi

L(bk_write_less32bytes):
	movl	%edi, %eax
	sub	%ecx, %edx
	sub	%ecx, %eax
	POP (%edi)
L(bk_write_less32bytes_2):
	BRANCH_TO_JMPTBL_ENTRY (L(table_48_bytes_bwd), %ecx, 4)

	CFI_PUSH (%edi)

	.p2align 4
L(bk_align):
	cmp	$8, %ecx
	jbe	L(bk_write_less32bytes)
	testl	$1, %edx
	/* We get here only if (EDX & 3 ) != 0 so if (EDX & 1) ==0,
	then	(EDX & 2) must be != 0.  */
	jz	L(bk_got2)
	sub	$1, %edi
	sub	$1, %ecx
	sub	$1, %edx
	movzbl	(%edi), %eax
	movb	%al, (%edx)

	testl	$2, %edx
	jz	L(bk_aligned_4)

L(bk_got2):
	sub	$2, %edi
	sub	$2, %ecx
	sub	$2, %edx
	movzwl	(%edi), %eax
	movw	%ax, (%edx)
	jmp	L(bk_aligned_4)

	.p2align 4
L(bk_write_more64bytes):
	/* Check alignment of last byte.  */
	testl	$15, %edx
	jz	L(bk_ssse3_cpy_pre)

/* EDX is aligned 4 bytes, but not 16 bytes.  */
L(bk_ssse3_align):
	sub	$4, %edi
	sub	$4, %ecx
	sub	$4, %edx
	movl	(%edi), %eax
	movl	%eax, (%edx)

	testl	$15, %edx
	jz	L(bk_ssse3_cpy_pre)

	sub	$4, %edi
	sub	$4, %ecx
	sub	$4, %edx
	movl	(%edi), %eax
	movl	%eax, (%edx)

	testl	$15, %edx
	jz	L(bk_ssse3_cpy_pre)

	sub	$4, %edi
	sub	$4, %ecx
	sub	$4, %edx
	movl	(%edi), %eax
	movl	%eax, (%edx)

L(bk_ssse3_cpy_pre):
	cmp	$64, %ecx
	jb	L(bk_write_more32bytes)

	.p2align 4
L(bk_ssse3_cpy):
	sub	$64, %edi
	sub	$64, %ecx
	sub	$64, %edx
	movdqu	0x30(%edi), %xmm3
	movdqa	%xmm3, 0x30(%edx)
	movdqu	0x20(%edi), %xmm2
	movdqa	%xmm2, 0x20(%edx)
	movdqu	0x10(%edi), %xmm1
	movdqa	%xmm1, 0x10(%edx)
	movdqu	(%edi), %xmm0
	movdqa	%xmm0, (%edx)
	cmp	$64, %ecx
	jae	L(bk_ssse3_cpy)
	jmp	L(bk_write_64bytesless)

# endif

END (MEMCPY)

#endif
