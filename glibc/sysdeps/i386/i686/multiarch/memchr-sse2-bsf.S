/* Optimized memchr with sse2
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

# define PARMS  4
# define STR1  PARMS
# define STR2  STR1+4

# ifndef USE_AS_RAWMEMCHR
#  define LEN   STR2+4
#  define RETURN  POP(%edi); ret; CFI_PUSH(%edi);
# endif

# ifndef MEMCHR
#  define MEMCHR __memchr_sse2_bsf
# endif

	.text
ENTRY (MEMCHR)

	mov	STR1(%esp), %ecx
	movd	STR2(%esp), %xmm1

# ifndef USE_AS_RAWMEMCHR
	mov	LEN(%esp), %edx
	test	%edx, %edx
	jz	L(return_null_1)
# endif
	mov	%ecx, %eax

	punpcklbw %xmm1, %xmm1
	punpcklbw %xmm1, %xmm1

	and	$63, %ecx
	pshufd	$0, %xmm1, %xmm1

	cmp	$48, %ecx
	ja	L(crosscache)

	movdqu	(%eax), %xmm0
	pcmpeqb	%xmm1, %xmm0
/* Check if there is a match.  */
	pmovmskb %xmm0, %ecx
	test	%ecx, %ecx
	je	L(unaligned_no_match_1)
/* Check which byte is a match.  */
	bsf	%ecx, %ecx

# ifndef USE_AS_RAWMEMCHR
	sub	%ecx, %edx
	jbe	L(return_null_1)
# endif
	add	%ecx, %eax
	ret

	.p2align 4
L(unaligned_no_match_1):
# ifndef USE_AS_RAWMEMCHR
	sub	$16, %edx
	jbe	L(return_null_1)
	PUSH	(%edi)
	lea	16(%eax), %edi
	and	$15, %eax
	and	$-16, %edi
	add	%eax, %edx
# else
	lea	16(%eax), %edx
	and	$-16, %edx
# endif
	jmp	L(loop_prolog)

	.p2align 4
L(return_null_1):
	xor	%eax, %eax
	ret

# ifndef USE_AS_RAWMEMCHR
	CFI_POP	(%edi)
# endif

	.p2align 4
L(crosscache):
/* Handle unaligned string.  */

# ifndef USE_AS_RAWMEMCHR
	PUSH	(%edi)
	mov	%eax, %edi
	and	$15, %ecx
	and	$-16, %edi
	movdqa	(%edi), %xmm0
# else
	mov	%eax, %edx
	and	$15, %ecx
	and	$-16, %edx
	movdqa	(%edx), %xmm0
# endif
	pcmpeqb	%xmm1, %xmm0
/* Check if there is a match.  */
	pmovmskb %xmm0, %eax
/* Remove the leading bytes.  */
	sar	%cl, %eax
	test	%eax, %eax
	je	L(unaligned_no_match)
/* Check which byte is a match.  */
	bsf	%eax, %eax

# ifndef USE_AS_RAWMEMCHR
	sub	%eax, %edx
	jbe	L(return_null)
	add	%edi, %eax
	add	%ecx, %eax
	RETURN
# else
	add	%edx, %eax
	add	%ecx, %eax
	ret
# endif

	.p2align 4
L(unaligned_no_match):
# ifndef USE_AS_RAWMEMCHR
        /* Calculate the last acceptable address and check for possible
           addition overflow by using satured math:
           edx = ecx + edx
           edx |= -(edx < ecx)  */
	add	%ecx, %edx
	sbb	%eax, %eax
	or	%eax, %edx
	sub	$16, %edx
	jbe	L(return_null)
	add	$16, %edi
# else
	add	$16, %edx
# endif

	.p2align 4
/* Loop start on aligned string.  */
L(loop_prolog):
# ifndef USE_AS_RAWMEMCHR
	sub	$64, %edx
	jbe	L(exit_loop)
	movdqa	(%edi), %xmm0
# else
	movdqa	(%edx), %xmm0
# endif
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)

# ifndef USE_AS_RAWMEMCHR
	movdqa	16(%edi), %xmm2
# else
	movdqa	16(%edx), %xmm2
# endif
	pcmpeqb	%xmm1, %xmm2
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

# ifndef USE_AS_RAWMEMCHR
	movdqa	32(%edi), %xmm3
# else
	movdqa	32(%edx), %xmm3
# endif
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32)

# ifndef USE_AS_RAWMEMCHR
	movdqa	48(%edi), %xmm4
# else
	movdqa	48(%edx), %xmm4
# endif
	pcmpeqb	%xmm1, %xmm4

# ifndef USE_AS_RAWMEMCHR
	add	$64, %edi
# else
	add	$64, %edx
# endif
	pmovmskb %xmm4, %eax
	test	%eax, %eax
	jnz	L(matches0)

# ifndef USE_AS_RAWMEMCHR
	test	$0x3f, %edi
# else
	test	$0x3f, %edx
# endif
	jz	L(align64_loop)

# ifndef USE_AS_RAWMEMCHR
	sub	$64, %edx
	jbe	L(exit_loop)
	movdqa	(%edi), %xmm0
# else
	movdqa	(%edx), %xmm0
# endif
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)

# ifndef USE_AS_RAWMEMCHR
	movdqa	16(%edi), %xmm2
# else
	movdqa	16(%edx), %xmm2
# endif
	pcmpeqb	%xmm1, %xmm2
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

# ifndef USE_AS_RAWMEMCHR
	movdqa	32(%edi), %xmm3
# else
	movdqa	32(%edx), %xmm3
# endif
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32)

# ifndef USE_AS_RAWMEMCHR
	movdqa	48(%edi), %xmm3
# else
	movdqa	48(%edx), %xmm3
# endif
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax

# ifndef USE_AS_RAWMEMCHR
	add	$64, %edi
# else
	add	$64, %edx
# endif
	test	%eax, %eax
	jnz	L(matches0)

# ifndef USE_AS_RAWMEMCHR
	mov	%edi, %ecx
	and	$-64, %edi
	and	$63, %ecx
	add	%ecx, %edx
# else
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
	pmovmskb %xmm4, %eax

# ifndef USE_AS_RAWMEMCHR
	add	$64, %edi
# else
	add	$64, %edx
# endif

	test	%eax, %eax
	jz	L(align64_loop)

# ifndef USE_AS_RAWMEMCHR
	sub	$64, %edi
# else
	sub	$64, %edx
# endif

	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)

	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

# ifndef USE_AS_RAWMEMCHR
	movdqa	32(%edi), %xmm3
# else
	movdqa	32(%edx), %xmm3
# endif

	pcmpeqb	%xmm1, %xmm3

# ifndef USE_AS_RAWMEMCHR
	pcmpeqb	48(%edi), %xmm1
# else
	pcmpeqb	48(%edx), %xmm1
# endif
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32)

	pmovmskb %xmm1, %eax
	bsf	%eax, %eax

# ifndef USE_AS_RAWMEMCHR
	lea	48(%edi, %eax), %eax
	RETURN
# else
	lea	48(%edx, %eax), %eax
	ret
# endif

# ifndef USE_AS_RAWMEMCHR
	.p2align 4
L(exit_loop):
	add	$64, %edx
	cmp	$32, %edx
	jbe	L(exit_loop_32)

	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)

	movdqa	16(%edi), %xmm2
	pcmpeqb	%xmm1, %xmm2
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

	movdqa	32(%edi), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32_1)
	cmp	$48, %edx
	jbe	L(return_null)

	pcmpeqb	48(%edi), %xmm1
	pmovmskb %xmm1, %eax
	test	%eax, %eax
	jnz	L(matches48_1)
	xor	%eax, %eax
	RETURN

	.p2align 4
L(exit_loop_32):
	movdqa	(%edi), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches_1)
	cmp	$16, %edx
	jbe	L(return_null)

	pcmpeqb	16(%edi), %xmm1
	pmovmskb %xmm1, %eax
	test	%eax, %eax
	jnz	L(matches16_1)
	xor	%eax, %eax
	RETURN
# endif
	.p2align 4
L(matches0):
	bsf	%eax, %eax
# ifndef USE_AS_RAWMEMCHR
	lea	-16(%eax, %edi), %eax
	RETURN
# else
	lea	-16(%eax, %edx), %eax
	ret
# endif

	.p2align 4
L(matches):
	bsf	%eax, %eax
# ifndef USE_AS_RAWMEMCHR
	add	%edi, %eax
	RETURN
# else
	add	%edx, %eax
	ret
# endif

	.p2align 4
L(matches16):
	bsf	%eax, %eax
# ifndef USE_AS_RAWMEMCHR
	lea	16(%eax, %edi), %eax
	RETURN
# else
	lea	16(%eax, %edx), %eax
	ret
# endif

	.p2align 4
L(matches32):
	bsf	%eax, %eax
# ifndef USE_AS_RAWMEMCHR
	lea	32(%eax, %edi), %eax
	RETURN
# else
	lea	32(%eax, %edx), %eax
	ret
# endif

# ifndef USE_AS_RAWMEMCHR
	.p2align 4
L(matches_1):
	bsf	%eax, %eax
	sub	%eax, %edx
	jbe	L(return_null)

	add	%edi, %eax
	RETURN

	.p2align 4
L(matches16_1):
	sub	$16, %edx
	bsf	%eax, %eax
	sub	%eax, %edx
	jbe	L(return_null)

	lea	16(%edi, %eax), %eax
	RETURN

	.p2align 4
L(matches32_1):
	sub	$32, %edx
	bsf	%eax, %eax
	sub	%eax, %edx
	jbe	L(return_null)

	lea	32(%edi, %eax), %eax
	RETURN

	.p2align 4
L(matches48_1):
	sub	$48, %edx
	bsf	%eax, %eax
	sub	%eax, %edx
	jbe	L(return_null)

	lea	48(%edi, %eax), %eax
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
