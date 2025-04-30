/* Optimized memchr with sse2 without bsf
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
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

# define CFI_PUSH(REG)	\
	cfi_adjust_cfa_offset (4);	\
	cfi_rel_offset (REG, 0)

# define CFI_POP(REG)	\
	cfi_adjust_cfa_offset (-4);	\
	cfi_restore (REG)

# define PUSH(REG) pushl REG; CFI_PUSH (REG)
# define POP(REG) popl REG; CFI_POP (REG)

# ifndef USE_AS_RAWMEMCHR
#  define ENTRANCE PUSH(%edi);
#  define PARMS  8
#  define RETURN  POP(%edi); ret; CFI_PUSH(%edi);
# else
#  define ENTRANCE
#  define PARMS  4
# endif

# define STR1  PARMS
# define STR2  STR1+4

# ifndef USE_AS_RAWMEMCHR
#  define LEN   STR2+4
# endif

# ifndef MEMCHR
#  define MEMCHR __memchr_sse2
# endif

	atom_text_section
ENTRY (MEMCHR)
	ENTRANCE
	mov	STR1(%esp), %ecx
	movd	STR2(%esp), %xmm1
# ifndef USE_AS_RAWMEMCHR
	mov	LEN(%esp), %edx
	test	%edx, %edx
	jz	L(return_null)
# endif

	punpcklbw %xmm1, %xmm1
# ifndef USE_AS_RAWMEMCHR
	mov	%ecx, %edi
# else
	mov	%ecx, %edx
# endif
	punpcklbw %xmm1, %xmm1

	and	$63, %ecx
	pshufd	$0, %xmm1, %xmm1
	cmp	$48, %ecx
	ja	L(crosscache)

# ifndef USE_AS_RAWMEMCHR
	movdqu	(%edi), %xmm0
# else
	movdqu	(%edx), %xmm0
# endif
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
# ifndef USE_AS_RAWMEMCHR
	jnz	L(match_case2_prolog)

	sub	$16, %edx
	jbe	L(return_null)
	lea	16(%edi), %edi
	and	$15, %ecx
	and	$-16, %edi
	add	%ecx, %edx
# else
	jnz	L(match_case1_prolog)
	lea	16(%edx), %edx
	and	$-16, %edx
# endif
	jmp	L(loop_prolog)

	.p2align 4
L(crosscache):
	and	$15, %ecx
# ifndef USE_AS_RAWMEMCHR
	and	$-16, %edi
	movdqa	(%edi), %xmm0
# else
	and	$-16, %edx
	movdqa	(%edx), %xmm0
# endif
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	sar	%cl, %eax
	test	%eax, %eax

# ifndef USE_AS_RAWMEMCHR
	jnz	L(match_case2_prolog1)
        /* "ecx" is less than 16.  Calculate "edx + ecx - 16" by using
	   "edx - (16 - ecx)" instead of "(edx + ecx) - 16" to void
	   possible addition overflow.  */
	neg	%ecx
	add	$16, %ecx
	sub	%ecx, %edx
	jbe	L(return_null)
	lea	16(%edi), %edi
# else
	jnz	L(match_case1_prolog1)
	lea	16(%edx), %edx
# endif

	.p2align 4
L(loop_prolog):
# ifndef USE_AS_RAWMEMCHR
	sub	$64, %edx
	jbe	L(exit_loop)
	movdqa	(%edi), %xmm0
# else
	movdqa	(%edx), %xmm0
# endif
	pcmpeqb	%xmm1, %xmm0
	xor	%ecx, %ecx
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(match_case1)

# ifndef USE_AS_RAWMEMCHR
	movdqa	16(%edi), %xmm2
# else
	movdqa	16(%edx), %xmm2
# endif
	pcmpeqb	%xmm1, %xmm2
	lea	16(%ecx), %ecx
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(match_case1)

# ifndef USE_AS_RAWMEMCHR
	movdqa	32(%edi), %xmm3
# else
	movdqa	32(%edx), %xmm3
# endif
	pcmpeqb	%xmm1, %xmm3
	lea	16(%ecx), %ecx
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(match_case1)

# ifndef USE_AS_RAWMEMCHR
	movdqa	48(%edi), %xmm4
# else
	movdqa	48(%edx), %xmm4
# endif
	pcmpeqb	%xmm1, %xmm4
	lea	16(%ecx), %ecx
	pmovmskb %xmm4, %eax
	test	%eax, %eax
	jnz	L(match_case1)

# ifndef USE_AS_RAWMEMCHR
	lea	64(%edi), %edi
	sub	$64, %edx
	jbe	L(exit_loop)

	movdqa	(%edi), %xmm0
# else
	lea	64(%edx), %edx
	movdqa	(%edx), %xmm0
# endif
	pcmpeqb	%xmm1, %xmm0
	xor	%ecx, %ecx
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(match_case1)

# ifndef USE_AS_RAWMEMCHR
	movdqa	16(%edi), %xmm2
# else
	movdqa	16(%edx), %xmm2
# endif
	pcmpeqb	%xmm1, %xmm2
	lea	16(%ecx), %ecx
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(match_case1)

# ifndef USE_AS_RAWMEMCHR
	movdqa	32(%edi), %xmm3
# else
	movdqa	32(%edx), %xmm3
# endif
	pcmpeqb	%xmm1, %xmm3
	lea	16(%ecx), %ecx
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(match_case1)

# ifndef USE_AS_RAWMEMCHR
	movdqa	48(%edi), %xmm4
# else
	movdqa	48(%edx), %xmm4
# endif
	pcmpeqb	%xmm1, %xmm4
	lea	16(%ecx), %ecx
	pmovmskb %xmm4, %eax
	test	%eax, %eax
	jnz	L(match_case1)

# ifndef USE_AS_RAWMEMCHR
	lea	64(%edi), %edi
	mov	%edi, %ecx
	and	$-64, %edi
	and	$63, %ecx
	add	%ecx, %edx
# else
	lea	64(%edx), %edx
	and	$-64, %edx
# endif

	.p2align 4
L(align64_loop):

# ifndef USE_AS_RAWMEMCHR
	sub	$64, %edx
	jbe	L(exit_loop)
	movdqa	(%edi), %xmm0
	movdqa	16(%edi), %xmm2
	movdqa	32(%edi), %xmm3
	movdqa	48(%edi), %xmm4
# else
	movdqa	(%edx), %xmm0
	movdqa	16(%edx), %xmm2
	movdqa	32(%edx), %xmm3
	movdqa	48(%edx), %xmm4
# endif
	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm1, %xmm2
	pcmpeqb	%xmm1, %xmm3
	pcmpeqb	%xmm1, %xmm4

	pmaxub	%xmm0, %xmm3
	pmaxub	%xmm2, %xmm4
	pmaxub	%xmm3, %xmm4
# ifndef USE_AS_RAWMEMCHR
	add	$64, %edi
# else
	add	$64, %edx
# endif
	pmovmskb %xmm4, %eax

	test	%eax, %eax
	jz	L(align64_loop)

# ifndef USE_AS_RAWMEMCHR
	sub	$64, %edi
# else
	sub	$64, %edx
# endif

	pmovmskb %xmm0, %eax
	xor	%ecx, %ecx
	test	%eax, %eax
	jnz	L(match_case1)

	pmovmskb %xmm2, %eax
	lea	16(%ecx), %ecx
	test	%eax, %eax
	jnz	L(match_case1)

# ifndef USE_AS_RAWMEMCHR
	movdqa	32(%edi), %xmm3
# else
	movdqa	32(%edx), %xmm3
# endif
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	lea	16(%ecx), %ecx
	test	%eax, %eax
	jnz	L(match_case1)

# ifndef USE_AS_RAWMEMCHR
	pcmpeqb	48(%edi), %xmm1
# else
	pcmpeqb	48(%edx), %xmm1
# endif
	pmovmskb %xmm1, %eax
	lea	16(%ecx), %ecx

	.p2align 4
L(match_case1):
# ifndef USE_AS_RAWMEMCHR
	add	%ecx, %edi
# else
L(match_case1_prolog1):
	add	%ecx, %edx
L(match_case1_prolog):
# endif
	test	%al, %al
	jz	L(match_case1_high)
	mov	%al, %cl
	and	$15, %cl
	jz	L(match_case1_8)
	test	$0x01, %al
	jnz	L(ExitCase1_1)
	test	$0x02, %al
	jnz	L(ExitCase1_2)
	test	$0x04, %al
	jnz	L(ExitCase1_3)
# ifndef USE_AS_RAWMEMCHR
	lea	3(%edi), %eax
	RETURN
# else
	lea	3(%edx), %eax
	ret
# endif

	.p2align 4
L(match_case1_8):
	test	$0x10, %al
	jnz	L(ExitCase1_5)
	test	$0x20, %al
	jnz	L(ExitCase1_6)
	test	$0x40, %al
	jnz	L(ExitCase1_7)
# ifndef USE_AS_RAWMEMCHR
	lea	7(%edi), %eax
	RETURN
# else
	lea	7(%edx), %eax
	ret
# endif

	.p2align 4
L(match_case1_high):
	mov	%ah, %ch
	and	$15, %ch
	jz	L(match_case1_high_8)
	test	$0x01, %ah
	jnz	L(ExitCase1_9)
	test	$0x02, %ah
	jnz	L(ExitCase1_10)
	test	$0x04, %ah
	jnz	L(ExitCase1_11)
# ifndef USE_AS_RAWMEMCHR
	lea	11(%edi), %eax
	RETURN
# else
	lea	11(%edx), %eax
	ret
# endif

	.p2align 4
L(match_case1_high_8):
	test	$0x10, %ah
	jnz	L(ExitCase1_13)
	test	$0x20, %ah
	jnz	L(ExitCase1_14)
	test	$0x40, %ah
	jnz	L(ExitCase1_15)
# ifndef USE_AS_RAWMEMCHR
	lea	15(%edi), %eax
	RETURN
# else
	lea	15(%edx), %eax
	ret
# endif

# ifndef USE_AS_RAWMEMCHR
	.p2align 4
L(exit_loop):
	add	$64, %edx

	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm1, %xmm0
	xor	%ecx, %ecx
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(match_case2)
	cmp	$16, %edx
	jbe	L(return_null)

	movdqa	16(%edi), %xmm2
	pcmpeqb	%xmm1, %xmm2
	lea	16(%ecx), %ecx
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(match_case2)
	cmp	$32, %edx
	jbe	L(return_null)

	movdqa	32(%edi), %xmm3
	pcmpeqb	%xmm1, %xmm3
	lea	16(%ecx), %ecx
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(match_case2)
	cmp	$48, %edx
	jbe	L(return_null)

	pcmpeqb	48(%edi), %xmm1
	lea	16(%ecx), %ecx
	pmovmskb %xmm1, %eax
	test	%eax, %eax
	jnz	L(match_case2)

	xor	%eax, %eax
	RETURN
# endif

	.p2align 4
L(ExitCase1_1):
# ifndef USE_AS_RAWMEMCHR
	mov	%edi, %eax
	RETURN
# else
	mov	%edx, %eax
	ret
# endif

	.p2align 4
L(ExitCase1_2):
# ifndef USE_AS_RAWMEMCHR
	lea	1(%edi), %eax
	RETURN
# else
	lea	1(%edx), %eax
	ret
# endif

	.p2align 4
L(ExitCase1_3):
# ifndef USE_AS_RAWMEMCHR
	lea	2(%edi), %eax
	RETURN
# else
	lea	2(%edx), %eax
	ret
# endif

	.p2align 4
L(ExitCase1_5):
# ifndef USE_AS_RAWMEMCHR
	lea	4(%edi), %eax
	RETURN
# else
	lea	4(%edx), %eax
	ret
# endif

	.p2align 4
L(ExitCase1_6):
# ifndef USE_AS_RAWMEMCHR
	lea	5(%edi), %eax
	RETURN
# else
	lea	5(%edx), %eax
	ret
# endif

	.p2align 4
L(ExitCase1_7):
# ifndef USE_AS_RAWMEMCHR
	lea	6(%edi), %eax
	RETURN
# else
	lea	6(%edx), %eax
	ret
# endif

	.p2align 4
L(ExitCase1_9):
# ifndef USE_AS_RAWMEMCHR
	lea	8(%edi), %eax
	RETURN
# else
	lea	8(%edx), %eax
	ret
# endif

	.p2align 4
L(ExitCase1_10):
# ifndef USE_AS_RAWMEMCHR
	lea	9(%edi), %eax
	RETURN
# else
	lea	9(%edx), %eax
	ret
# endif

	.p2align 4
L(ExitCase1_11):
# ifndef USE_AS_RAWMEMCHR
	lea	10(%edi), %eax
	RETURN
# else
	lea	10(%edx), %eax
	ret
# endif

	.p2align 4
L(ExitCase1_13):
# ifndef USE_AS_RAWMEMCHR
	lea	12(%edi), %eax
	RETURN
# else
	lea	12(%edx), %eax
	ret
# endif

	.p2align 4
L(ExitCase1_14):
# ifndef USE_AS_RAWMEMCHR
	lea	13(%edi), %eax
	RETURN
# else
	lea	13(%edx), %eax
	ret
# endif

	.p2align 4
L(ExitCase1_15):
# ifndef USE_AS_RAWMEMCHR
	lea	14(%edi), %eax
	RETURN
# else
	lea	14(%edx), %eax
	ret
# endif

# ifndef USE_AS_RAWMEMCHR
	.p2align 4
L(match_case2):
	sub	%ecx, %edx
L(match_case2_prolog1):
	add	%ecx, %edi
L(match_case2_prolog):
	test	%al, %al
	jz	L(match_case2_high)
	mov	%al, %cl
	and	$15, %cl
	jz	L(match_case2_8)
	test	$0x01, %al
	jnz	L(ExitCase2_1)
	test	$0x02, %al
	jnz	L(ExitCase2_2)
	test	$0x04, %al
	jnz	L(ExitCase2_3)
	sub	$4, %edx
	jb	L(return_null)
	lea	3(%edi), %eax
	RETURN

	.p2align 4
L(match_case2_8):
	test	$0x10, %al
	jnz	L(ExitCase2_5)
	test	$0x20, %al
	jnz	L(ExitCase2_6)
	test	$0x40, %al
	jnz	L(ExitCase2_7)
	sub	$8, %edx
	jb	L(return_null)
	lea	7(%edi), %eax
	RETURN

	.p2align 4
L(match_case2_high):
	mov	%ah, %ch
	and	$15, %ch
	jz	L(match_case2_high_8)
	test	$0x01, %ah
	jnz	L(ExitCase2_9)
	test	$0x02, %ah
	jnz	L(ExitCase2_10)
	test	$0x04, %ah
	jnz	L(ExitCase2_11)
	sub	$12, %edx
	jb	L(return_null)
	lea	11(%edi), %eax
	RETURN

	.p2align 4
L(match_case2_high_8):
	test	$0x10, %ah
	jnz	L(ExitCase2_13)
	test	$0x20, %ah
	jnz	L(ExitCase2_14)
	test	$0x40, %ah
	jnz	L(ExitCase2_15)
	sub	$16, %edx
	jb	L(return_null)
	lea	15(%edi), %eax
	RETURN

	.p2align 4
L(ExitCase2_1):
	mov	%edi, %eax
	RETURN

	.p2align 4
L(ExitCase2_2):
	sub	$2, %edx
	jb	L(return_null)
	lea	1(%edi), %eax
	RETURN

	.p2align 4
L(ExitCase2_3):
	sub	$3, %edx
	jb	L(return_null)
	lea	2(%edi), %eax
	RETURN

	.p2align 4
L(ExitCase2_5):
	sub	$5, %edx
	jb	L(return_null)
	lea	4(%edi), %eax
	RETURN

	.p2align 4
L(ExitCase2_6):
	sub	$6, %edx
	jb	L(return_null)
	lea	5(%edi), %eax
	RETURN

	.p2align 4
L(ExitCase2_7):
	sub	$7, %edx
	jb	L(return_null)
	lea	6(%edi), %eax
	RETURN

	.p2align 4
L(ExitCase2_9):
	sub	$9, %edx
	jb	L(return_null)
	lea	8(%edi), %eax
	RETURN

	.p2align 4
L(ExitCase2_10):
	sub	$10, %edx
	jb	L(return_null)
	lea	9(%edi), %eax
	RETURN

	.p2align 4
L(ExitCase2_11):
	sub	$11, %edx
	jb	L(return_null)
	lea	10(%edi), %eax
	RETURN

	.p2align 4
L(ExitCase2_13):
	sub	$13, %edx
	jb	L(return_null)
	lea	12(%edi), %eax
	RETURN

	.p2align 4
L(ExitCase2_14):
	sub	$14, %edx
	jb	L(return_null)
	lea	13(%edi), %eax
	RETURN

	.p2align 4
L(ExitCase2_15):
	sub	$15, %edx
	jb	L(return_null)
	lea	14(%edi), %eax
	RETURN
# endif

	.p2align 4
L(return_null):
	xor	%eax, %eax
# ifndef USE_AS_RAWMEMCHR
	RETURN
# else
	ret
# endif

END (MEMCHR)
#endif
