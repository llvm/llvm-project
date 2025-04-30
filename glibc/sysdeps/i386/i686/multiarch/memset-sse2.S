/* memset with SSE2
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

#if IS_IN (libc)

#include <sysdep.h>
#include "asm-syntax.h"

#define CFI_PUSH(REG)						\
  cfi_adjust_cfa_offset (4);					\
  cfi_rel_offset (REG, 0)

#define CFI_POP(REG)						\
  cfi_adjust_cfa_offset (-4);					\
  cfi_restore (REG)

#define PUSH(REG)	pushl REG; CFI_PUSH (REG)
#define POP(REG)	popl REG; CFI_POP (REG)

#ifdef USE_AS_BZERO
# define DEST		PARMS
# define LEN		DEST+4
# define SETRTNVAL
#else
# define DEST		PARMS
# define CHR		DEST+4
# define LEN		CHR+4
# define SETRTNVAL	movl DEST(%esp), %eax
#endif

#ifdef PIC
# define ENTRANCE	PUSH (%ebx);
# define RETURN_END	POP (%ebx); ret
# define RETURN		RETURN_END; CFI_PUSH (%ebx)
# define PARMS		8		/* Preserve EBX.  */
# define JMPTBL(I, B)	I - B

/* Load an entry in a jump table into EBX and branch to it.  TABLE is a
   jump table with relative offsets.   */
# define BRANCH_TO_JMPTBL_ENTRY(TABLE)				\
    /* We first load PC into EBX.  */				\
    SETUP_PIC_REG(bx);						\
    /* Get the address of the jump table.  */			\
    add		$(TABLE - .), %ebx;				\
    /* Get the entry and convert the relative offset to the	\
       absolute address.  */					\
    add		(%ebx,%ecx,4), %ebx;				\
    add		%ecx, %edx;					\
    /* We loaded the jump table and adjusted EDX. Go.  */	\
    _CET_NOTRACK jmp	*%ebx
#else
# define ENTRANCE
# define RETURN_END	ret
# define RETURN		RETURN_END
# define PARMS		4
# define JMPTBL(I, B)	I

/* Branch to an entry in a jump table.  TABLE is a jump table with
   absolute offsets.  */
# define BRANCH_TO_JMPTBL_ENTRY(TABLE)				\
    add		%ecx, %edx;					\
    _CET_NOTRACK jmp *TABLE(,%ecx,4)
#endif

	.section .text.sse2,"ax",@progbits
#if defined SHARED && IS_IN (libc) && !defined USE_AS_BZERO
ENTRY (__memset_chk_sse2)
	movl	12(%esp), %eax
	cmpl	%eax, 16(%esp)
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (__memset_chk_sse2)
#endif
ENTRY (__memset_sse2)
	ENTRANCE

	movl	LEN(%esp), %ecx
#ifdef USE_AS_BZERO
	xor	%eax, %eax
#else
	movzbl	CHR(%esp), %eax
	movb	%al, %ah
	/* Fill the whole EAX with pattern.  */
	movl	%eax, %edx
	shl	$16, %eax
	or	%edx, %eax
#endif
	movl	DEST(%esp), %edx
	cmp	$32, %ecx
	jae	L(32bytesormore)

L(write_less32bytes):
	BRANCH_TO_JMPTBL_ENTRY (L(table_less_32bytes))


	.pushsection .rodata.sse2,"a",@progbits
	ALIGN (2)
L(table_less_32bytes):
	.int	JMPTBL (L(write_0bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_1bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_2bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_3bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_4bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_5bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_6bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_7bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_8bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_9bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_10bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_11bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_12bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_13bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_14bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_15bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_16bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_17bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_18bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_19bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_20bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_21bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_22bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_23bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_24bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_25bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_26bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_27bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_28bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_29bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_30bytes), L(table_less_32bytes))
	.int	JMPTBL (L(write_31bytes), L(table_less_32bytes))
	.popsection

	ALIGN (4)
L(write_28bytes):
	movl	%eax, -28(%edx)
L(write_24bytes):
	movl	%eax, -24(%edx)
L(write_20bytes):
	movl	%eax, -20(%edx)
L(write_16bytes):
	movl	%eax, -16(%edx)
L(write_12bytes):
	movl	%eax, -12(%edx)
L(write_8bytes):
	movl	%eax, -8(%edx)
L(write_4bytes):
	movl	%eax, -4(%edx)
L(write_0bytes):
	SETRTNVAL
	RETURN

	ALIGN (4)
L(write_29bytes):
	movl	%eax, -29(%edx)
L(write_25bytes):
	movl	%eax, -25(%edx)
L(write_21bytes):
	movl	%eax, -21(%edx)
L(write_17bytes):
	movl	%eax, -17(%edx)
L(write_13bytes):
	movl	%eax, -13(%edx)
L(write_9bytes):
	movl	%eax, -9(%edx)
L(write_5bytes):
	movl	%eax, -5(%edx)
L(write_1bytes):
	movb	%al, -1(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(write_30bytes):
	movl	%eax, -30(%edx)
L(write_26bytes):
	movl	%eax, -26(%edx)
L(write_22bytes):
	movl	%eax, -22(%edx)
L(write_18bytes):
	movl	%eax, -18(%edx)
L(write_14bytes):
	movl	%eax, -14(%edx)
L(write_10bytes):
	movl	%eax, -10(%edx)
L(write_6bytes):
	movl	%eax, -6(%edx)
L(write_2bytes):
	movw	%ax, -2(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(write_31bytes):
	movl	%eax, -31(%edx)
L(write_27bytes):
	movl	%eax, -27(%edx)
L(write_23bytes):
	movl	%eax, -23(%edx)
L(write_19bytes):
	movl	%eax, -19(%edx)
L(write_15bytes):
	movl	%eax, -15(%edx)
L(write_11bytes):
	movl	%eax, -11(%edx)
L(write_7bytes):
	movl	%eax, -7(%edx)
L(write_3bytes):
	movw	%ax, -3(%edx)
	movb	%al, -1(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
/* ECX > 32 and EDX is 4 byte aligned.  */
L(32bytesormore):
	/* Fill xmm0 with the pattern.  */
#ifdef USE_AS_BZERO
	pxor	%xmm0, %xmm0
#else
	movd	%eax, %xmm0
	pshufd	$0, %xmm0, %xmm0
#endif
	testl	$0xf, %edx
	jz	L(aligned_16)
/* ECX > 32 and EDX is not 16 byte aligned.  */
L(not_aligned_16):
	movdqu	%xmm0, (%edx)
	movl	%edx, %eax
	and	$-16, %edx
	add	$16, %edx
	sub	%edx, %eax
	add	%eax, %ecx
	movd	%xmm0, %eax

	ALIGN (4)
L(aligned_16):
	cmp	$128, %ecx
	jae	L(128bytesormore)

L(aligned_16_less128bytes):
	BRANCH_TO_JMPTBL_ENTRY (L(table_16_128bytes))

	ALIGN (4)
L(128bytesormore):
#ifdef SHARED_CACHE_SIZE
	PUSH (%ebx)
	mov	$SHARED_CACHE_SIZE, %ebx
#else
# ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	mov	__x86_shared_cache_size@GOTOFF(%ebx), %ebx
# else
	PUSH (%ebx)
	mov	__x86_shared_cache_size, %ebx
# endif
#endif
	cmp	%ebx, %ecx
	jae	L(128bytesormore_nt_start)


#ifdef DATA_CACHE_SIZE
	POP (%ebx)
# define RESTORE_EBX_STATE CFI_PUSH (%ebx)
	cmp	$DATA_CACHE_SIZE, %ecx
#else
# ifdef PIC
#  define RESTORE_EBX_STATE
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_data_cache_size@GOTOFF(%ebx), %ecx
# else
	POP (%ebx)
#  define RESTORE_EBX_STATE CFI_PUSH (%ebx)
	cmp	__x86_data_cache_size, %ecx
# endif
#endif

	jae	L(128bytes_L2_normal)
	subl	$128, %ecx
L(128bytesormore_normal):
	sub	$128, %ecx
	movdqa	%xmm0, (%edx)
	movdqa	%xmm0, 0x10(%edx)
	movdqa	%xmm0, 0x20(%edx)
	movdqa	%xmm0, 0x30(%edx)
	movdqa	%xmm0, 0x40(%edx)
	movdqa	%xmm0, 0x50(%edx)
	movdqa	%xmm0, 0x60(%edx)
	movdqa	%xmm0, 0x70(%edx)
	lea	128(%edx), %edx
	jb	L(128bytesless_normal)


	sub	$128, %ecx
	movdqa	%xmm0, (%edx)
	movdqa	%xmm0, 0x10(%edx)
	movdqa	%xmm0, 0x20(%edx)
	movdqa	%xmm0, 0x30(%edx)
	movdqa	%xmm0, 0x40(%edx)
	movdqa	%xmm0, 0x50(%edx)
	movdqa	%xmm0, 0x60(%edx)
	movdqa	%xmm0, 0x70(%edx)
	lea	128(%edx), %edx
	jae	L(128bytesormore_normal)

L(128bytesless_normal):
	add	$128, %ecx
	BRANCH_TO_JMPTBL_ENTRY (L(table_16_128bytes))

	ALIGN (4)
L(128bytes_L2_normal):
	prefetcht0	0x380(%edx)
	prefetcht0	0x3c0(%edx)
	sub	$128, %ecx
	movdqa	%xmm0, (%edx)
	movaps	%xmm0, 0x10(%edx)
	movaps	%xmm0, 0x20(%edx)
	movaps	%xmm0, 0x30(%edx)
	movaps	%xmm0, 0x40(%edx)
	movaps	%xmm0, 0x50(%edx)
	movaps	%xmm0, 0x60(%edx)
	movaps	%xmm0, 0x70(%edx)
	add	$128, %edx
	cmp	$128, %ecx
	jae	L(128bytes_L2_normal)

L(128bytesless_L2_normal):
	BRANCH_TO_JMPTBL_ENTRY (L(table_16_128bytes))

	RESTORE_EBX_STATE
L(128bytesormore_nt_start):
	sub	%ebx, %ecx
	ALIGN (4)
L(128bytesormore_shared_cache_loop):
	prefetcht0	0x3c0(%edx)
	prefetcht0	0x380(%edx)
	sub	$0x80, %ebx
	movdqa	%xmm0, (%edx)
	movdqa	%xmm0, 0x10(%edx)
	movdqa	%xmm0, 0x20(%edx)
	movdqa	%xmm0, 0x30(%edx)
	movdqa	%xmm0, 0x40(%edx)
	movdqa	%xmm0, 0x50(%edx)
	movdqa	%xmm0, 0x60(%edx)
	movdqa	%xmm0, 0x70(%edx)
	add	$0x80, %edx
	cmp	$0x80, %ebx
	jae	L(128bytesormore_shared_cache_loop)
	cmp	$0x80, %ecx
	jb	L(shared_cache_loop_end)
	ALIGN (4)
L(128bytesormore_nt):
	sub	$0x80, %ecx
	movntdq	%xmm0, (%edx)
	movntdq	%xmm0, 0x10(%edx)
	movntdq	%xmm0, 0x20(%edx)
	movntdq	%xmm0, 0x30(%edx)
	movntdq	%xmm0, 0x40(%edx)
	movntdq	%xmm0, 0x50(%edx)
	movntdq	%xmm0, 0x60(%edx)
	movntdq	%xmm0, 0x70(%edx)
	add	$0x80, %edx
	cmp	$0x80, %ecx
	jae	L(128bytesormore_nt)
	sfence
L(shared_cache_loop_end):
#if defined DATA_CACHE_SIZE || !defined PIC
	POP (%ebx)
#endif
	BRANCH_TO_JMPTBL_ENTRY (L(table_16_128bytes))


	.pushsection .rodata.sse2,"a",@progbits
	ALIGN (2)
L(table_16_128bytes):
	.int	JMPTBL (L(aligned_16_0bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_1bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_2bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_3bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_4bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_5bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_6bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_7bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_8bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_9bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_10bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_11bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_12bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_13bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_14bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_15bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_16bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_17bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_18bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_19bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_20bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_21bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_22bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_23bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_24bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_25bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_26bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_27bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_28bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_29bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_30bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_31bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_32bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_33bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_34bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_35bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_36bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_37bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_38bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_39bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_40bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_41bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_42bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_43bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_44bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_45bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_46bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_47bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_48bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_49bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_50bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_51bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_52bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_53bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_54bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_55bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_56bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_57bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_58bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_59bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_60bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_61bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_62bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_63bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_64bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_65bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_66bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_67bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_68bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_69bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_70bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_71bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_72bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_73bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_74bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_75bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_76bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_77bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_78bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_79bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_80bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_81bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_82bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_83bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_84bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_85bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_86bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_87bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_88bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_89bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_90bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_91bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_92bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_93bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_94bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_95bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_96bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_97bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_98bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_99bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_100bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_101bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_102bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_103bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_104bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_105bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_106bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_107bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_108bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_109bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_110bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_111bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_112bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_113bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_114bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_115bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_116bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_117bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_118bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_119bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_120bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_121bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_122bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_123bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_124bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_125bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_126bytes), L(table_16_128bytes))
	.int	JMPTBL (L(aligned_16_127bytes), L(table_16_128bytes))
	.popsection

	ALIGN (4)
L(aligned_16_112bytes):
	movdqa	%xmm0, -112(%edx)
L(aligned_16_96bytes):
	movdqa	%xmm0, -96(%edx)
L(aligned_16_80bytes):
	movdqa	%xmm0, -80(%edx)
L(aligned_16_64bytes):
	movdqa	%xmm0, -64(%edx)
L(aligned_16_48bytes):
	movdqa	%xmm0, -48(%edx)
L(aligned_16_32bytes):
	movdqa	%xmm0, -32(%edx)
L(aligned_16_16bytes):
	movdqa	%xmm0, -16(%edx)
L(aligned_16_0bytes):
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_113bytes):
	movdqa	%xmm0, -113(%edx)
L(aligned_16_97bytes):
	movdqa	%xmm0, -97(%edx)
L(aligned_16_81bytes):
	movdqa	%xmm0, -81(%edx)
L(aligned_16_65bytes):
	movdqa	%xmm0, -65(%edx)
L(aligned_16_49bytes):
	movdqa	%xmm0, -49(%edx)
L(aligned_16_33bytes):
	movdqa	%xmm0, -33(%edx)
L(aligned_16_17bytes):
	movdqa	%xmm0, -17(%edx)
L(aligned_16_1bytes):
	movb	%al, -1(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_114bytes):
	movdqa	%xmm0, -114(%edx)
L(aligned_16_98bytes):
	movdqa	%xmm0, -98(%edx)
L(aligned_16_82bytes):
	movdqa	%xmm0, -82(%edx)
L(aligned_16_66bytes):
	movdqa	%xmm0, -66(%edx)
L(aligned_16_50bytes):
	movdqa	%xmm0, -50(%edx)
L(aligned_16_34bytes):
	movdqa	%xmm0, -34(%edx)
L(aligned_16_18bytes):
	movdqa	%xmm0, -18(%edx)
L(aligned_16_2bytes):
	movw	%ax, -2(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_115bytes):
	movdqa	%xmm0, -115(%edx)
L(aligned_16_99bytes):
	movdqa	%xmm0, -99(%edx)
L(aligned_16_83bytes):
	movdqa	%xmm0, -83(%edx)
L(aligned_16_67bytes):
	movdqa	%xmm0, -67(%edx)
L(aligned_16_51bytes):
	movdqa	%xmm0, -51(%edx)
L(aligned_16_35bytes):
	movdqa	%xmm0, -35(%edx)
L(aligned_16_19bytes):
	movdqa	%xmm0, -19(%edx)
L(aligned_16_3bytes):
	movw	%ax, -3(%edx)
	movb	%al, -1(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_116bytes):
	movdqa	%xmm0, -116(%edx)
L(aligned_16_100bytes):
	movdqa	%xmm0, -100(%edx)
L(aligned_16_84bytes):
	movdqa	%xmm0, -84(%edx)
L(aligned_16_68bytes):
	movdqa	%xmm0, -68(%edx)
L(aligned_16_52bytes):
	movdqa	%xmm0, -52(%edx)
L(aligned_16_36bytes):
	movdqa	%xmm0, -36(%edx)
L(aligned_16_20bytes):
	movdqa	%xmm0, -20(%edx)
L(aligned_16_4bytes):
	movl	%eax, -4(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_117bytes):
	movdqa	%xmm0, -117(%edx)
L(aligned_16_101bytes):
	movdqa	%xmm0, -101(%edx)
L(aligned_16_85bytes):
	movdqa	%xmm0, -85(%edx)
L(aligned_16_69bytes):
	movdqa	%xmm0, -69(%edx)
L(aligned_16_53bytes):
	movdqa	%xmm0, -53(%edx)
L(aligned_16_37bytes):
	movdqa	%xmm0, -37(%edx)
L(aligned_16_21bytes):
	movdqa	%xmm0, -21(%edx)
L(aligned_16_5bytes):
	movl	%eax, -5(%edx)
	movb	%al, -1(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_118bytes):
	movdqa	%xmm0, -118(%edx)
L(aligned_16_102bytes):
	movdqa	%xmm0, -102(%edx)
L(aligned_16_86bytes):
	movdqa	%xmm0, -86(%edx)
L(aligned_16_70bytes):
	movdqa	%xmm0, -70(%edx)
L(aligned_16_54bytes):
	movdqa	%xmm0, -54(%edx)
L(aligned_16_38bytes):
	movdqa	%xmm0, -38(%edx)
L(aligned_16_22bytes):
	movdqa	%xmm0, -22(%edx)
L(aligned_16_6bytes):
	movl	%eax, -6(%edx)
	movw	%ax, -2(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_119bytes):
	movdqa	%xmm0, -119(%edx)
L(aligned_16_103bytes):
	movdqa	%xmm0, -103(%edx)
L(aligned_16_87bytes):
	movdqa	%xmm0, -87(%edx)
L(aligned_16_71bytes):
	movdqa	%xmm0, -71(%edx)
L(aligned_16_55bytes):
	movdqa	%xmm0, -55(%edx)
L(aligned_16_39bytes):
	movdqa	%xmm0, -39(%edx)
L(aligned_16_23bytes):
	movdqa	%xmm0, -23(%edx)
L(aligned_16_7bytes):
	movl	%eax, -7(%edx)
	movw	%ax, -3(%edx)
	movb	%al, -1(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_120bytes):
	movdqa	%xmm0, -120(%edx)
L(aligned_16_104bytes):
	movdqa	%xmm0, -104(%edx)
L(aligned_16_88bytes):
	movdqa	%xmm0, -88(%edx)
L(aligned_16_72bytes):
	movdqa	%xmm0, -72(%edx)
L(aligned_16_56bytes):
	movdqa	%xmm0, -56(%edx)
L(aligned_16_40bytes):
	movdqa	%xmm0, -40(%edx)
L(aligned_16_24bytes):
	movdqa	%xmm0, -24(%edx)
L(aligned_16_8bytes):
	movq	%xmm0, -8(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_121bytes):
	movdqa	%xmm0, -121(%edx)
L(aligned_16_105bytes):
	movdqa	%xmm0, -105(%edx)
L(aligned_16_89bytes):
	movdqa	%xmm0, -89(%edx)
L(aligned_16_73bytes):
	movdqa	%xmm0, -73(%edx)
L(aligned_16_57bytes):
	movdqa	%xmm0, -57(%edx)
L(aligned_16_41bytes):
	movdqa	%xmm0, -41(%edx)
L(aligned_16_25bytes):
	movdqa	%xmm0, -25(%edx)
L(aligned_16_9bytes):
	movq	%xmm0, -9(%edx)
	movb	%al, -1(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_122bytes):
	movdqa	%xmm0, -122(%edx)
L(aligned_16_106bytes):
	movdqa	%xmm0, -106(%edx)
L(aligned_16_90bytes):
	movdqa	%xmm0, -90(%edx)
L(aligned_16_74bytes):
	movdqa	%xmm0, -74(%edx)
L(aligned_16_58bytes):
	movdqa	%xmm0, -58(%edx)
L(aligned_16_42bytes):
	movdqa	%xmm0, -42(%edx)
L(aligned_16_26bytes):
	movdqa	%xmm0, -26(%edx)
L(aligned_16_10bytes):
	movq	%xmm0, -10(%edx)
	movw	%ax, -2(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_123bytes):
	movdqa	%xmm0, -123(%edx)
L(aligned_16_107bytes):
	movdqa	%xmm0, -107(%edx)
L(aligned_16_91bytes):
	movdqa	%xmm0, -91(%edx)
L(aligned_16_75bytes):
	movdqa	%xmm0, -75(%edx)
L(aligned_16_59bytes):
	movdqa	%xmm0, -59(%edx)
L(aligned_16_43bytes):
	movdqa	%xmm0, -43(%edx)
L(aligned_16_27bytes):
	movdqa	%xmm0, -27(%edx)
L(aligned_16_11bytes):
	movq	%xmm0, -11(%edx)
	movw	%ax, -3(%edx)
	movb	%al, -1(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_124bytes):
	movdqa	%xmm0, -124(%edx)
L(aligned_16_108bytes):
	movdqa	%xmm0, -108(%edx)
L(aligned_16_92bytes):
	movdqa	%xmm0, -92(%edx)
L(aligned_16_76bytes):
	movdqa	%xmm0, -76(%edx)
L(aligned_16_60bytes):
	movdqa	%xmm0, -60(%edx)
L(aligned_16_44bytes):
	movdqa	%xmm0, -44(%edx)
L(aligned_16_28bytes):
	movdqa	%xmm0, -28(%edx)
L(aligned_16_12bytes):
	movq	%xmm0, -12(%edx)
	movl	%eax, -4(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_125bytes):
	movdqa	%xmm0, -125(%edx)
L(aligned_16_109bytes):
	movdqa	%xmm0, -109(%edx)
L(aligned_16_93bytes):
	movdqa	%xmm0, -93(%edx)
L(aligned_16_77bytes):
	movdqa	%xmm0, -77(%edx)
L(aligned_16_61bytes):
	movdqa	%xmm0, -61(%edx)
L(aligned_16_45bytes):
	movdqa	%xmm0, -45(%edx)
L(aligned_16_29bytes):
	movdqa	%xmm0, -29(%edx)
L(aligned_16_13bytes):
	movq	%xmm0, -13(%edx)
	movl	%eax, -5(%edx)
	movb	%al, -1(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_126bytes):
	movdqa	%xmm0, -126(%edx)
L(aligned_16_110bytes):
	movdqa	%xmm0, -110(%edx)
L(aligned_16_94bytes):
	movdqa	%xmm0, -94(%edx)
L(aligned_16_78bytes):
	movdqa	%xmm0, -78(%edx)
L(aligned_16_62bytes):
	movdqa	%xmm0, -62(%edx)
L(aligned_16_46bytes):
	movdqa	%xmm0, -46(%edx)
L(aligned_16_30bytes):
	movdqa	%xmm0, -30(%edx)
L(aligned_16_14bytes):
	movq	%xmm0, -14(%edx)
	movl	%eax, -6(%edx)
	movw	%ax, -2(%edx)
	SETRTNVAL
	RETURN

	ALIGN (4)
L(aligned_16_127bytes):
	movdqa	%xmm0, -127(%edx)
L(aligned_16_111bytes):
	movdqa	%xmm0, -111(%edx)
L(aligned_16_95bytes):
	movdqa	%xmm0, -95(%edx)
L(aligned_16_79bytes):
	movdqa	%xmm0, -79(%edx)
L(aligned_16_63bytes):
	movdqa	%xmm0, -63(%edx)
L(aligned_16_47bytes):
	movdqa	%xmm0, -47(%edx)
L(aligned_16_31bytes):
	movdqa	%xmm0, -31(%edx)
L(aligned_16_15bytes):
	movq	%xmm0, -15(%edx)
	movl	%eax, -7(%edx)
	movw	%ax, -3(%edx)
	movb	%al, -1(%edx)
	SETRTNVAL
	RETURN_END

END (__memset_sse2)

#endif
