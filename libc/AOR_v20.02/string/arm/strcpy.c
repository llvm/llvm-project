/*
 * strcpy
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#if defined (__thumb2__) && !defined (__thumb__)

/* For GLIBC:
#include <string.h>
#include <memcopy.h>

#undef strcmp
*/

#ifdef __thumb2__
#define magic1(REG) "#0x01010101"
#define magic2(REG) "#0x80808080"
#else
#define magic1(REG) #REG
#define magic2(REG) #REG ", lsl #7"
#endif

char* __attribute__((naked))
__strcpy_arm (char* dst, const char* src)
{
  __asm__ (
       "pld	[r1, #0]\n\t"
       "eor	r2, r0, r1\n\t"
       "mov	ip, r0\n\t"
       "tst	r2, #3\n\t"
       "bne	4f\n\t"
       "tst	r1, #3\n\t"
       "bne	3f\n"
  "5:\n\t"
# ifndef __thumb2__
       "str	r5, [sp, #-4]!\n\t"
       "mov	r5, #0x01\n\t"
       "orr	r5, r5, r5, lsl #8\n\t"
       "orr	r5, r5, r5, lsl #16\n\t"
# endif

       "str	r4, [sp, #-4]!\n\t"
       "tst	r1, #4\n\t"
       "ldr	r3, [r1], #4\n\t"
       "beq	2f\n\t"
       "sub	r2, r3, "magic1(r5)"\n\t"
       "bics	r2, r2, r3\n\t"
       "tst	r2, "magic2(r5)"\n\t"
       "itt	eq\n\t"
       "streq	r3, [ip], #4\n\t"
       "ldreq	r3, [r1], #4\n"
       "bne	1f\n\t"
       /* Inner loop.  We now know that r1 is 64-bit aligned, so we
	  can safely fetch up to two words.  This allows us to avoid
	  load stalls.  */
       ".p2align 2\n"
  "2:\n\t"
       "pld	[r1, #8]\n\t"
       "ldr	r4, [r1], #4\n\t"
       "sub	r2, r3, "magic1(r5)"\n\t"
       "bics	r2, r2, r3\n\t"
       "tst	r2, "magic2(r5)"\n\t"
       "sub	r2, r4, "magic1(r5)"\n\t"
       "bne	1f\n\t"
       "str	r3, [ip], #4\n\t"
       "bics	r2, r2, r4\n\t"
       "tst	r2, "magic2(r5)"\n\t"
       "itt	eq\n\t"
       "ldreq	r3, [r1], #4\n\t"
       "streq	r4, [ip], #4\n\t"
       "beq	2b\n\t"
       "mov	r3, r4\n"
  "1:\n\t"
# ifdef __ARMEB__
       "rors	r3, r3, #24\n\t"
# endif
       "strb	r3, [ip], #1\n\t"
       "tst	r3, #0xff\n\t"
# ifdef __ARMEL__
       "ror	r3, r3, #8\n\t"
# endif
       "bne	1b\n\t"
       "ldr	r4, [sp], #4\n\t"
# ifndef __thumb2__
       "ldr	r5, [sp], #4\n\t"
# endif
       "BX LR\n"

       /* Strings have the same offset from word alignment, but it's
	  not zero.  */
  "3:\n\t"
       "tst	r1, #1\n\t"
       "beq	1f\n\t"
       "ldrb	r2, [r1], #1\n\t"
       "strb	r2, [ip], #1\n\t"
       "cmp	r2, #0\n\t"
       "it	eq\n"
       "BXEQ LR\n"
  "1:\n\t"
       "tst	r1, #2\n\t"
       "beq	5b\n\t"
       "ldrh	r2, [r1], #2\n\t"
# ifdef __ARMEB__
       "tst	r2, #0xff00\n\t"
       "iteet	ne\n\t"
       "strneh	r2, [ip], #2\n\t"
       "lsreq	r2, r2, #8\n\t"
       "streqb	r2, [ip]\n\t"
       "tstne	r2, #0xff\n\t"
# else
       "tst	r2, #0xff\n\t"
       "itet	ne\n\t"
       "strneh	r2, [ip], #2\n\t"
       "streqb	r2, [ip]\n\t"
       "tstne	r2, #0xff00\n\t"
# endif
       "bne	5b\n\t"
       "BX LR\n"

       /* src and dst do not have a common word-alignment.  Fall back to
	  byte copying.  */
  "4:\n\t"
       "ldrb	r2, [r1], #1\n\t"
       "strb	r2, [ip], #1\n\t"
       "cmp	r2, #0\n\t"
       "bne	4b\n\t"
       "BX LR");
}
/* For GLIBC: libc_hidden_builtin_def (strcpy) */

#endif /* defined (__thumb2__) && !defined (__thumb__)  */
