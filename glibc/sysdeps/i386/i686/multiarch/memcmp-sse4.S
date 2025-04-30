/* memcmp with SSE4.2, wmemcmp with SSE4.2
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

# include <sysdep.h>

# ifndef MEMCMP
#  define MEMCMP	__memcmp_sse4_2
# endif

# define CFI_PUSH(REG)	\
	cfi_adjust_cfa_offset (4);	\
	cfi_rel_offset (REG, 0)

# define CFI_POP(REG)	\
	cfi_adjust_cfa_offset (-4);	\
	cfi_restore (REG)

# define PUSH(REG)	pushl REG; CFI_PUSH (REG)
# define POP(REG)	popl REG; CFI_POP (REG)

# define PARMS	4
# define BLK1	PARMS
# define BLK2	BLK1 + 4
# define LEN	BLK2 + 4
# define RETURN	POP (%ebx); ret; CFI_PUSH (%ebx)


# ifdef PIC
#  define JMPTBL(I, B)	I - B

/* Load an entry in a jump table into EBX and branch to it.  TABLE is a
	jump	table with relative offsets.  INDEX is a register contains the
	index	into the jump table.   SCALE is the scale of INDEX. */

#  define BRANCH_TO_JMPTBL_ENTRY(TABLE, INDEX, SCALE)	\
/* We first load PC into EBX.  */	\
	SETUP_PIC_REG(bx);	\
/* Get the address of the jump table.  */	\
	addl	$(TABLE - .), %ebx;	\
/* Get the entry and convert the relative offset to the	\
	absolute	address.  */	\
	addl	(%ebx,INDEX,SCALE), %ebx;	\
/* We loaded the jump table and adjusted EDX/ESI. Go.  */	\
	_CET_NOTRACK jmp *%ebx
# else
#  define JMPTBL(I, B)	I

/* Load an entry in a jump table into EBX and branch to it.  TABLE is a
	jump	table with relative offsets.  INDEX is a register contains the
	index	into the jump table.   SCALE is the scale of INDEX. */
#  define BRANCH_TO_JMPTBL_ENTRY(TABLE, INDEX, SCALE)	\
	_CET_NOTRACK jmp *TABLE(,INDEX,SCALE)
# endif


/* Warning!
           wmemcmp has to use SIGNED comparison for elements.
           memcmp has to use UNSIGNED comparison for elemnts.
*/

	.section .text.sse4.2,"ax",@progbits
ENTRY (MEMCMP)
	movl	BLK1(%esp), %eax
	movl	BLK2(%esp), %edx
	movl	LEN(%esp), %ecx

# ifdef USE_AS_WMEMCMP
	shl	$2, %ecx
	test	%ecx, %ecx
	jz	L(return0)
# else
	cmp	$1, %ecx
	jbe	L(less1bytes)
# endif

	pxor	%xmm0, %xmm0
	cmp	$64, %ecx
	ja	L(64bytesormore)
	cmp	$8, %ecx

# ifndef USE_AS_WMEMCMP
	PUSH	(%ebx)
	jb	L(less8bytes)
# else
	jb	L(less8bytes)
	PUSH	(%ebx)
# endif

	add	%ecx, %edx
	add	%ecx, %eax
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %ecx, 4)

# ifndef USE_AS_WMEMCMP
	.p2align 4
L(less8bytes):
	mov	(%eax), %bl
	cmpb	(%edx), %bl
	jne	L(nonzero)

	mov	1(%eax), %bl
	cmpb	1(%edx), %bl
	jne	L(nonzero)

	cmp	$2, %ecx
	jz	L(0bytes)

	mov	2(%eax), %bl
	cmpb	2(%edx), %bl
	jne	L(nonzero)

	cmp	$3, %ecx
	jz	L(0bytes)

	mov	3(%eax), %bl
	cmpb	3(%edx), %bl
	jne	L(nonzero)

	cmp	$4, %ecx
	jz	L(0bytes)

	mov	4(%eax), %bl
	cmpb	4(%edx), %bl
	jne	L(nonzero)

	cmp	$5, %ecx
	jz	L(0bytes)

	mov	5(%eax), %bl
	cmpb	5(%edx), %bl
	jne	L(nonzero)

	cmp	$6, %ecx
	jz	L(0bytes)

	mov	6(%eax), %bl
	cmpb	6(%edx), %bl
	je	L(0bytes)

L(nonzero):
	POP	(%ebx)
	mov	$1, %eax
	ja	L(above)
	neg	%eax
L(above):
	ret
	CFI_PUSH (%ebx)
# endif

	.p2align 4
L(0bytes):
	POP	(%ebx)
	xor	%eax, %eax
	ret

# ifdef USE_AS_WMEMCMP

/* for wmemcmp, case N == 1 */

	.p2align 4
L(less8bytes):
	mov	(%eax), %ecx
	cmp	(%edx), %ecx
	je	L(return0)
	mov	$1, %eax
	jg	L(find_diff_bigger)
	neg	%eax
	ret

	.p2align 4
L(find_diff_bigger):
	ret

	.p2align 4
L(return0):
	xor	%eax, %eax
	ret
# endif

# ifndef USE_AS_WMEMCMP
	.p2align 4
L(less1bytes):
	jb	L(0bytesend)
	movzbl	(%eax), %eax
	movzbl	(%edx), %edx
	sub	%edx, %eax
	ret

	.p2align 4
L(0bytesend):
	xor	%eax, %eax
	ret
# endif
	.p2align 4
L(64bytesormore):
	PUSH	(%ebx)
	mov	%ecx, %ebx
	mov	$64, %ecx
	sub	$64, %ebx
L(64bytesormore_loop):
	movdqu	(%eax), %xmm1
	movdqu	(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(find_16diff)

	movdqu	16(%eax), %xmm1
	movdqu	16(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(find_32diff)

	movdqu	32(%eax), %xmm1
	movdqu	32(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(find_48diff)

	movdqu	48(%eax), %xmm1
	movdqu	48(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(find_64diff)
	add	%ecx, %eax
	add	%ecx, %edx
	sub	%ecx, %ebx
	jae	L(64bytesormore_loop)
	add	%ebx, %ecx
	add	%ecx, %edx
	add	%ecx, %eax
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %ecx, 4)

# ifdef USE_AS_WMEMCMP

/* Label needs only for table_64bytes filling */
L(unreal_case):
/* no code here */

# endif
	.p2align 4
L(find_16diff):
	sub	$16, %ecx
L(find_32diff):
	sub	$16, %ecx
L(find_48diff):
	sub	$16, %ecx
L(find_64diff):
	add	%ecx, %edx
	add	%ecx, %eax

# ifndef USE_AS_WMEMCMP
	.p2align 4
L(16bytes):
	mov	-16(%eax), %ecx
	mov	-16(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(12bytes):
	mov	-12(%eax), %ecx
	mov	-12(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(8bytes):
	mov	-8(%eax), %ecx
	mov	-8(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(4bytes):
	mov	-4(%eax), %ecx
	mov	-4(%edx), %ebx
	cmp	%ebx, %ecx
	mov	$0, %eax
	jne	L(find_diff)
	RETURN
# else
	.p2align 4
L(16bytes):
	mov	-16(%eax), %ecx
	cmp	-16(%edx), %ecx
	jne	L(find_diff)
L(12bytes):
	mov	-12(%eax), %ecx
	cmp	-12(%edx), %ecx
	jne	L(find_diff)
L(8bytes):
	mov	-8(%eax), %ecx
	cmp	-8(%edx), %ecx
	jne	L(find_diff)
L(4bytes):
	mov	-4(%eax), %ecx
	cmp	-4(%edx), %ecx
	mov	$0, %eax
	jne	L(find_diff)
	RETURN
# endif

# ifndef USE_AS_WMEMCMP
	.p2align 4
L(49bytes):
	movdqu	-49(%eax), %xmm1
	movdqu	-49(%edx), %xmm2
	mov	$-49, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(33bytes):
	movdqu	-33(%eax), %xmm1
	movdqu	-33(%edx), %xmm2
	mov	$-33, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(17bytes):
	mov	-17(%eax), %ecx
	mov	-17(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(13bytes):
	mov	-13(%eax), %ecx
	mov	-13(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(9bytes):
	mov	-9(%eax), %ecx
	mov	-9(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(5bytes):
	mov	-5(%eax), %ecx
	mov	-5(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movzbl	-1(%eax), %ecx
	cmp	-1(%edx), %cl
	mov	$0, %eax
	jne	L(end)
	RETURN

	.p2align 4
L(50bytes):
	mov	$-50, %ebx
	movdqu	-50(%eax), %xmm1
	movdqu	-50(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(34bytes):
	mov	$-34, %ebx
	movdqu	-34(%eax), %xmm1
	movdqu	-34(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(18bytes):
	mov	-18(%eax), %ecx
	mov	-18(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(14bytes):
	mov	-14(%eax), %ecx
	mov	-14(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(10bytes):
	mov	-10(%eax), %ecx
	mov	-10(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(6bytes):
	mov	-6(%eax), %ecx
	mov	-6(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(2bytes):
	movzwl	-2(%eax), %ecx
	movzwl	-2(%edx), %ebx
	cmp	%bl, %cl
	jne	L(end)
	cmp	%bh, %ch
	mov	$0, %eax
	jne	L(end)
	RETURN

	.p2align 4
L(51bytes):
	mov	$-51, %ebx
	movdqu	-51(%eax), %xmm1
	movdqu	-51(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(35bytes):
	mov	$-35, %ebx
	movdqu	-35(%eax), %xmm1
	movdqu	-35(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(19bytes):
	movl	-19(%eax), %ecx
	movl	-19(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(15bytes):
	movl	-15(%eax), %ecx
	movl	-15(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(11bytes):
	movl	-11(%eax), %ecx
	movl	-11(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(7bytes):
	movl	-7(%eax), %ecx
	movl	-7(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
L(3bytes):
	movzwl	-3(%eax), %ecx
	movzwl	-3(%edx), %ebx
	cmpb	%bl, %cl
	jne	L(end)
	cmp	%bx, %cx
	jne	L(end)
L(1bytes):
	movzbl	-1(%eax), %eax
	cmpb	-1(%edx), %al
	mov	$0, %eax
	jne	L(end)
	RETURN
# endif
	.p2align 4
L(52bytes):
	movdqu	-52(%eax), %xmm1
	movdqu	-52(%edx), %xmm2
	mov	$-52, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(36bytes):
	movdqu	-36(%eax), %xmm1
	movdqu	-36(%edx), %xmm2
	mov	$-36, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(20bytes):
	movdqu	-20(%eax), %xmm1
	movdqu	-20(%edx), %xmm2
	mov	$-20, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-4(%eax), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-4(%edx), %ebx
	cmp	%ebx, %ecx
# else
	cmp	-4(%edx), %ecx
# endif
	mov	$0, %eax
	jne	L(find_diff)
	RETURN

# ifndef USE_AS_WMEMCMP
	.p2align 4
L(53bytes):
	movdqu	-53(%eax), %xmm1
	movdqu	-53(%edx), %xmm2
	mov	$-53, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(37bytes):
	mov	$-37, %ebx
	movdqu	-37(%eax), %xmm1
	movdqu	-37(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(21bytes):
	mov	$-21, %ebx
	movdqu	-21(%eax), %xmm1
	movdqu	-21(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-5(%eax), %ecx
	mov	-5(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movzbl	-1(%eax), %ecx
	cmp	-1(%edx), %cl
	mov	$0, %eax
	jne	L(end)
	RETURN

	.p2align 4
L(54bytes):
	movdqu	-54(%eax), %xmm1
	movdqu	-54(%edx), %xmm2
	mov	$-54, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(38bytes):
	mov	$-38, %ebx
	movdqu	-38(%eax), %xmm1
	movdqu	-38(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(22bytes):
	mov	$-22, %ebx
	movdqu	-22(%eax), %xmm1
	movdqu	-22(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)

	mov	-6(%eax), %ecx
	mov	-6(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movzwl	-2(%eax), %ecx
	movzwl	-2(%edx), %ebx
	cmp	%bl, %cl
	jne	L(end)
	cmp	%bh, %ch
	mov	$0, %eax
	jne	L(end)
	RETURN

	.p2align 4
L(55bytes):
	movdqu	-55(%eax), %xmm1
	movdqu	-55(%edx), %xmm2
	mov	$-55, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(39bytes):
	mov	$-39, %ebx
	movdqu	-39(%eax), %xmm1
	movdqu	-39(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(23bytes):
	mov	$-23, %ebx
	movdqu	-23(%eax), %xmm1
	movdqu	-23(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	movl	-7(%eax), %ecx
	movl	-7(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movzwl	-3(%eax), %ecx
	movzwl	-3(%edx), %ebx
	cmpb	%bl, %cl
	jne	L(end)
	cmp	%bx, %cx
	jne	L(end)
	movzbl	-1(%eax), %eax
	cmpb	-1(%edx), %al
	mov	$0, %eax
	jne	L(end)
	RETURN
# endif
	.p2align 4
L(56bytes):
	movdqu	-56(%eax), %xmm1
	movdqu	-56(%edx), %xmm2
	mov	$-56, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(40bytes):
	mov	$-40, %ebx
	movdqu	-40(%eax), %xmm1
	movdqu	-40(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(24bytes):
	mov	$-24, %ebx
	movdqu	-24(%eax), %xmm1
	movdqu	-24(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)

	mov	-8(%eax), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-8(%edx), %ebx
	cmp	%ebx, %ecx
# else
	cmp	-8(%edx), %ecx
# endif
	jne	L(find_diff)

	mov	-4(%eax), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-4(%edx), %ebx
	cmp	%ebx, %ecx
# else
	cmp	-4(%edx), %ecx
# endif
	mov	$0, %eax
	jne	L(find_diff)
	RETURN

# ifndef USE_AS_WMEMCMP
	.p2align 4
L(57bytes):
	movdqu	-57(%eax), %xmm1
	movdqu	-57(%edx), %xmm2
	mov	$-57, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(41bytes):
	mov	$-41, %ebx
	movdqu	-41(%eax), %xmm1
	movdqu	-41(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(25bytes):
	mov	$-25, %ebx
	movdqu	-25(%eax), %xmm1
	movdqu	-25(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-9(%eax), %ecx
	mov	-9(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	mov	-5(%eax), %ecx
	mov	-5(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movzbl	-1(%eax), %ecx
	cmp	-1(%edx), %cl
	mov	$0, %eax
	jne	L(end)
	RETURN

	.p2align 4
L(58bytes):
	movdqu	-58(%eax), %xmm1
	movdqu	-58(%edx), %xmm2
	mov	$-58, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(42bytes):
	mov	$-42, %ebx
	movdqu	-42(%eax), %xmm1
	movdqu	-42(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(26bytes):
	mov	$-26, %ebx
	movdqu	-26(%eax), %xmm1
	movdqu	-26(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)

	mov	-10(%eax), %ecx
	mov	-10(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)

	mov	-6(%eax), %ecx
	mov	-6(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)

	movzwl	-2(%eax), %ecx
	movzwl	-2(%edx), %ebx
	cmp	%bl, %cl
	jne	L(end)
	cmp	%bh, %ch
	mov	$0, %eax
	jne	L(end)
	RETURN

	.p2align 4
L(59bytes):
	movdqu	-59(%eax), %xmm1
	movdqu	-59(%edx), %xmm2
	mov	$-59, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(43bytes):
	mov	$-43, %ebx
	movdqu	-43(%eax), %xmm1
	movdqu	-43(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(27bytes):
	mov	$-27, %ebx
	movdqu	-27(%eax), %xmm1
	movdqu	-27(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	movl	-11(%eax), %ecx
	movl	-11(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movl	-7(%eax), %ecx
	movl	-7(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movzwl	-3(%eax), %ecx
	movzwl	-3(%edx), %ebx
	cmpb	%bl, %cl
	jne	L(end)
	cmp	%bx, %cx
	jne	L(end)
	movzbl	-1(%eax), %eax
	cmpb	-1(%edx), %al
	mov	$0, %eax
	jne	L(end)
	RETURN
# endif
	.p2align 4
L(60bytes):
	movdqu	-60(%eax), %xmm1
	movdqu	-60(%edx), %xmm2
	mov	$-60, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(44bytes):
	mov	$-44, %ebx
	movdqu	-44(%eax), %xmm1
	movdqu	-44(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(28bytes):
	mov	$-28, %ebx
	movdqu	-28(%eax), %xmm1
	movdqu	-28(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)

	mov	-12(%eax), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-12(%edx), %ebx
	cmp	%ebx, %ecx
# else
	cmp	-12(%edx), %ecx
# endif
	jne	L(find_diff)

	mov	-8(%eax), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-8(%edx), %ebx
	cmp	%ebx, %ecx
# else
	cmp	-8(%edx), %ecx
# endif
	jne	L(find_diff)

	mov	-4(%eax), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-4(%edx), %ebx
	cmp	%ebx, %ecx
# else
	cmp	-4(%edx), %ecx
# endif
	mov	$0, %eax
	jne	L(find_diff)
	RETURN

# ifndef USE_AS_WMEMCMP
	.p2align 4
L(61bytes):
	movdqu	-61(%eax), %xmm1
	movdqu	-61(%edx), %xmm2
	mov	$-61, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(45bytes):
	mov	$-45, %ebx
	movdqu	-45(%eax), %xmm1
	movdqu	-45(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(29bytes):
	mov	$-29, %ebx
	movdqu	-29(%eax), %xmm1
	movdqu	-29(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)

	mov	-13(%eax), %ecx
	mov	-13(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)

	mov	-9(%eax), %ecx
	mov	-9(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)

	mov	-5(%eax), %ecx
	mov	-5(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movzbl	-1(%eax), %ecx
	cmp	-1(%edx), %cl
	mov	$0, %eax
	jne	L(end)
	RETURN

	.p2align 4
L(62bytes):
	movdqu	-62(%eax), %xmm1
	movdqu	-62(%edx), %xmm2
	mov	$-62, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(46bytes):
	mov	$-46, %ebx
	movdqu	-46(%eax), %xmm1
	movdqu	-46(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(30bytes):
	mov	$-30, %ebx
	movdqu	-30(%eax), %xmm1
	movdqu	-30(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-14(%eax), %ecx
	mov	-14(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	mov	-10(%eax), %ecx
	mov	-10(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	mov	-6(%eax), %ecx
	mov	-6(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movzwl	-2(%eax), %ecx
	movzwl	-2(%edx), %ebx
	cmp	%bl, %cl
	jne	L(end)
	cmp	%bh, %ch
	mov	$0, %eax
	jne	L(end)
	RETURN

	.p2align 4
L(63bytes):
	movdqu	-63(%eax), %xmm1
	movdqu	-63(%edx), %xmm2
	mov	$-63, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(47bytes):
	mov	$-47, %ebx
	movdqu	-47(%eax), %xmm1
	movdqu	-47(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(31bytes):
	mov	$-31, %ebx
	movdqu	-31(%eax), %xmm1
	movdqu	-31(%edx), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)

	movl	-15(%eax), %ecx
	movl	-15(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movl	-11(%eax), %ecx
	movl	-11(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movl	-7(%eax), %ecx
	movl	-7(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)
	movzwl	-3(%eax), %ecx
	movzwl	-3(%edx), %ebx
	cmpb	%bl, %cl
	jne	L(end)
	cmp	%bx, %cx
	jne	L(end)
	movzbl	-1(%eax), %eax
	cmpb	-1(%edx), %al
	mov	$0, %eax
	jne	L(end)
	RETURN
# endif

	.p2align 4
L(64bytes):
	movdqu	-64(%eax), %xmm1
	movdqu	-64(%edx), %xmm2
	mov	$-64, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(48bytes):
	movdqu	-48(%eax), %xmm1
	movdqu	-48(%edx), %xmm2
	mov	$-48, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(32bytes):
	movdqu	-32(%eax), %xmm1
	movdqu	-32(%edx), %xmm2
	mov	$-32, %ebx
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)

	mov	-16(%eax), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-16(%edx), %ebx
	cmp	%ebx, %ecx
# else
	cmp	-16(%edx), %ecx
# endif
	jne	L(find_diff)

	mov	-12(%eax), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-12(%edx), %ebx
	cmp	%ebx, %ecx
# else
	cmp	-12(%edx), %ecx
# endif
	jne	L(find_diff)

	mov	-8(%eax), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-8(%edx), %ebx
	cmp	%ebx, %ecx
# else
	cmp	-8(%edx), %ecx
# endif
	jne	L(find_diff)

	mov	-4(%eax), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-4(%edx), %ebx
	cmp	%ebx, %ecx
# else
	cmp	-4(%edx), %ecx
# endif
	mov	$0, %eax
	jne	L(find_diff)
	RETURN

# ifndef USE_AS_WMEMCMP
	.p2align 4
L(less16bytes):
	add	%ebx, %eax
	add	%ebx, %edx

	mov	(%eax), %ecx
	mov	(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)

	mov	4(%eax), %ecx
	mov	4(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)

	mov	8(%eax), %ecx
	mov	8(%edx), %ebx
	cmp	%ebx, %ecx
	jne	L(find_diff)

	mov	12(%eax), %ecx
	mov	12(%edx), %ebx
	cmp	%ebx, %ecx
	mov	$0, %eax
	jne	L(find_diff)
	RETURN
# else
	.p2align 4
L(less16bytes):
	add	%ebx, %eax
	add	%ebx, %edx

	mov	(%eax), %ecx
	cmp	(%edx), %ecx
	jne	L(find_diff)

	mov	4(%eax), %ecx
	cmp	4(%edx), %ecx
	jne	L(find_diff)

	mov	8(%eax), %ecx
	cmp	8(%edx), %ecx
	jne	L(find_diff)

	mov	12(%eax), %ecx
	cmp	12(%edx), %ecx

	mov	$0, %eax
	jne	L(find_diff)
	RETURN
# endif

	.p2align 4
L(find_diff):
# ifndef USE_AS_WMEMCMP
	cmpb	%bl, %cl
	jne	L(end)
	cmp	%bx, %cx
	jne	L(end)
	shr	$16,%ecx
	shr	$16,%ebx
	cmp	%bl, %cl
	jne	L(end)
	cmp	%bx, %cx
L(end):
	POP	(%ebx)
	mov	$1, %eax
	ja	L(bigger)
	neg	%eax
L(bigger):
	ret
# else
	POP	(%ebx)
	mov	$1, %eax
	jg	L(bigger)
	neg	%eax
	ret

	.p2align 4
L(bigger):
	ret
# endif
END (MEMCMP)

	.section .rodata.sse4.2,"a",@progbits
	.p2align 2
	.type	L(table_64bytes), @object
# ifndef USE_AS_WMEMCMP
L(table_64bytes):
	.int	JMPTBL (L(0bytes), L(table_64bytes))
	.int	JMPTBL (L(1bytes), L(table_64bytes))
	.int	JMPTBL (L(2bytes), L(table_64bytes))
	.int	JMPTBL (L(3bytes), L(table_64bytes))
	.int	JMPTBL (L(4bytes), L(table_64bytes))
	.int	JMPTBL (L(5bytes), L(table_64bytes))
	.int	JMPTBL (L(6bytes), L(table_64bytes))
	.int	JMPTBL (L(7bytes), L(table_64bytes))
	.int	JMPTBL (L(8bytes), L(table_64bytes))
	.int	JMPTBL (L(9bytes), L(table_64bytes))
	.int	JMPTBL (L(10bytes), L(table_64bytes))
	.int	JMPTBL (L(11bytes), L(table_64bytes))
	.int	JMPTBL (L(12bytes), L(table_64bytes))
	.int	JMPTBL (L(13bytes), L(table_64bytes))
	.int	JMPTBL (L(14bytes), L(table_64bytes))
	.int	JMPTBL (L(15bytes), L(table_64bytes))
	.int	JMPTBL (L(16bytes), L(table_64bytes))
	.int	JMPTBL (L(17bytes), L(table_64bytes))
	.int	JMPTBL (L(18bytes), L(table_64bytes))
	.int	JMPTBL (L(19bytes), L(table_64bytes))
	.int	JMPTBL (L(20bytes), L(table_64bytes))
	.int	JMPTBL (L(21bytes), L(table_64bytes))
	.int	JMPTBL (L(22bytes), L(table_64bytes))
	.int	JMPTBL (L(23bytes), L(table_64bytes))
	.int	JMPTBL (L(24bytes), L(table_64bytes))
	.int	JMPTBL (L(25bytes), L(table_64bytes))
	.int	JMPTBL (L(26bytes), L(table_64bytes))
	.int	JMPTBL (L(27bytes), L(table_64bytes))
	.int	JMPTBL (L(28bytes), L(table_64bytes))
	.int	JMPTBL (L(29bytes), L(table_64bytes))
	.int	JMPTBL (L(30bytes), L(table_64bytes))
	.int	JMPTBL (L(31bytes), L(table_64bytes))
	.int	JMPTBL (L(32bytes), L(table_64bytes))
	.int	JMPTBL (L(33bytes), L(table_64bytes))
	.int	JMPTBL (L(34bytes), L(table_64bytes))
	.int	JMPTBL (L(35bytes), L(table_64bytes))
	.int	JMPTBL (L(36bytes), L(table_64bytes))
	.int	JMPTBL (L(37bytes), L(table_64bytes))
	.int	JMPTBL (L(38bytes), L(table_64bytes))
	.int	JMPTBL (L(39bytes), L(table_64bytes))
	.int	JMPTBL (L(40bytes), L(table_64bytes))
	.int	JMPTBL (L(41bytes), L(table_64bytes))
	.int	JMPTBL (L(42bytes), L(table_64bytes))
	.int	JMPTBL (L(43bytes), L(table_64bytes))
	.int	JMPTBL (L(44bytes), L(table_64bytes))
	.int	JMPTBL (L(45bytes), L(table_64bytes))
	.int	JMPTBL (L(46bytes), L(table_64bytes))
	.int	JMPTBL (L(47bytes), L(table_64bytes))
	.int	JMPTBL (L(48bytes), L(table_64bytes))
	.int	JMPTBL (L(49bytes), L(table_64bytes))
	.int	JMPTBL (L(50bytes), L(table_64bytes))
	.int	JMPTBL (L(51bytes), L(table_64bytes))
	.int	JMPTBL (L(52bytes), L(table_64bytes))
	.int	JMPTBL (L(53bytes), L(table_64bytes))
	.int	JMPTBL (L(54bytes), L(table_64bytes))
	.int	JMPTBL (L(55bytes), L(table_64bytes))
	.int	JMPTBL (L(56bytes), L(table_64bytes))
	.int	JMPTBL (L(57bytes), L(table_64bytes))
	.int	JMPTBL (L(58bytes), L(table_64bytes))
	.int	JMPTBL (L(59bytes), L(table_64bytes))
	.int	JMPTBL (L(60bytes), L(table_64bytes))
	.int	JMPTBL (L(61bytes), L(table_64bytes))
	.int	JMPTBL (L(62bytes), L(table_64bytes))
	.int	JMPTBL (L(63bytes), L(table_64bytes))
	.int	JMPTBL (L(64bytes), L(table_64bytes))
# else
L(table_64bytes):
	.int	JMPTBL (L(0bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(4bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(8bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(12bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(16bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(20bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(24bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(28bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(32bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(36bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(40bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(44bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(48bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(52bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(56bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(60bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(64bytes), L(table_64bytes))
# endif
#endif
