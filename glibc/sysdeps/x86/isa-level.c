/* ELF program property for x86 ISA level.
   Copyright (C) 2020 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   In addition to the permissions in the GNU Lesser General Public
   License, the Free Software Foundation gives you unlimited
   permission to link the compiled version of this file with other
   programs, and to distribute those programs without any restriction
   coming from the use of this file.  (The Lesser General Public
   License restrictions do apply in other respects; for example, they
   cover modification of the file, and distribution when not linked
   into another program.)

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <elf.h>

/* ELF program property for x86 ISA level.  */
#ifdef INCLUDE_X86_ISA_LEVEL
# if defined __SSE__ && defined __SSE2__
/* NB: ISAs, excluding MMX, in x86-64 ISA level baseline are used.  */
#  define ISA_BASELINE	GNU_PROPERTY_X86_ISA_1_BASELINE
# else
#  define ISA_BASELINE	0
# endif

# if ISA_BASELINE && defined __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 \
     && defined HAVE_X86_LAHF_SAHF && defined __POPCNT__ \
     && defined __SSE3__ && defined __SSSE3__ && defined __SSE4_1__ \
     && defined __SSE4_2__
/* NB: ISAs in x86-64 ISA level v2 are used.  */
#  define ISA_V2	GNU_PROPERTY_X86_ISA_1_V2
# else
#  define ISA_V2	0
# endif

# if ISA_V2 && defined __AVX__ && defined __AVX2__ && defined __F16C__ \
     && defined __FMA__ && defined __LZCNT__ && defined HAVE_X86_MOVBE
/* NB: ISAs in x86-64 ISA level v3 are used.  */
#  define ISA_V3	GNU_PROPERTY_X86_ISA_1_V3
# else
#  define ISA_V3	0
# endif

# if ISA_V3 && defined __AVX512F__ && defined __AVX512BW__ \
     && defined __AVX512CD__ && defined __AVX512DQ__ \
     && defined __AVX512VL__
/* NB: ISAs in x86-64 ISA level v4 are used.  */
#  define ISA_V4	GNU_PROPERTY_X86_ISA_1_V4
# else
#  define ISA_V4	0
# endif

# ifndef ISA_LEVEL
#  define ISA_LEVEL (ISA_BASELINE | ISA_V2 | ISA_V3 | ISA_V4)
# endif

# if ISA_LEVEL
#  ifdef __LP64__
#   define PROPERTY_ALIGN 3
#  else
#   define PROPERTY_ALIGN 2
#  endif

#  define note_stringify(arg) note_stringify_1(arg)
#  define note_stringify_1(arg) #arg

asm(".pushsection \".note.gnu.property\",\"a\",@note\n"
"	.p2align " note_stringify (PROPERTY_ALIGN)
	/* name length.  */
"\n	.long 1f - 0f\n"
	/* data length.  */
"	.long 4f - 1f\n"
	/* note type: NT_GNU_PROPERTY_TYPE_0.  */
"	.long " note_stringify (NT_GNU_PROPERTY_TYPE_0)
	/* vendor name.  */
"\n0:	.asciz \"GNU\"\n"
"1:	.p2align " note_stringify (PROPERTY_ALIGN)
	/* pr_type: GNU_PROPERTY_X86_ISA_1_NEEDED.  */
"\n	.long " note_stringify (GNU_PROPERTY_X86_ISA_1_NEEDED)
	/* pr_datasz.  */
"\n	.long 3f - 2f\n"
	/* GNU_PROPERTY_X86_ISA_1_V[234].  */
"2:\n	 .long " note_stringify (ISA_LEVEL)
"\n3:\n	.p2align " note_stringify (PROPERTY_ALIGN)
"\n4:\n .popsection");
# endif /* ISA_LEVEL */
#endif /* INCLUDE_X86_ISA_LEVEL */
