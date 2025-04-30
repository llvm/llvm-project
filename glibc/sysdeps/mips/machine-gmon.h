/* Machine-specific calling sequence for `mcount' profiling function.  MIPS
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sgidefs.h>

#define _MCOUNT_DECL(frompc,selfpc) \
static void __attribute_used__ __mcount (u_long frompc, u_long selfpc)

/* Call __mcount with the return PC for our caller,
   and the return PC our caller will return to.  */

#if _MIPS_SIM == _ABIO32

#ifdef __PIC__
# define CPLOAD ".cpload $25;"
# define CPRESTORE ".cprestore 44\n\t"
#else
# define CPLOAD
# define CPRESTORE
#endif

#define MCOUNT asm(\
	".globl _mcount;\n\t" \
	".align 2;\n\t" \
	".set push;\n\t" \
	".set nomips16;\n\t" \
	".type _mcount,@function;\n\t" \
	".ent _mcount\n\t" \
        "_mcount:\n\t" \
        ".frame $sp,44,$31\n\t" \
        ".set noreorder;\n\t" \
        ".set noat;\n\t" \
        CPLOAD \
	"subu $29,$29,48;\n\t" \
	CPRESTORE \
        "sw $4,24($29);\n\t" \
        "sw $5,28($29);\n\t" \
        "sw $6,32($29);\n\t" \
        "sw $7,36($29);\n\t" \
        "sw $2,40($29);\n\t" \
        "sw $1,16($29);\n\t" \
        "sw $31,20($29);\n\t" \
        "move $5,$31;\n\t" \
        "move $4,$1;\n\t" \
        "jal __mcount;\n\t" \
        "nop;\n\t" \
        "lw $4,24($29);\n\t" \
        "lw $5,28($29);\n\t" \
        "lw $6,32($29);\n\t" \
        "lw $7,36($29);\n\t" \
        "lw $2,40($29);\n\t" \
        "lw $31,20($29);\n\t" \
        "lw $1,16($29);\n\t" \
        "addu $29,$29,56;\n\t" \
        "j $31;\n\t" \
        "move $31,$1;\n\t" \
	".end _mcount;\n\t" \
	".set pop");

#else

#ifdef __PIC__
# define CPSETUP ".cpsetup $25, 88, _mcount;"
# define CPRETURN ".cpreturn;"
#else
# define CPSETUP
# define CPRETURN
#endif

#if _MIPS_SIM == _ABIN32
# if !defined __mips_isa_rev || __mips_isa_rev < 6
#  define PTR_ADDU_STRING "add" /* no u */
#  define PTR_SUBU_STRING "sub" /* no u */
# else
#  define PTR_ADDU_STRING "addu"
#  define PTR_SUBU_STRING "subu"
# endif
#elif _MIPS_SIM == _ABI64
# define PTR_ADDU_STRING "daddu"
# define PTR_SUBU_STRING "dsubu"
#else
# error "Unknown ABI"
#endif

#define MCOUNT asm(\
	".globl _mcount;\n\t" \
	".align 3;\n\t" \
	".set push;\n\t" \
	".set nomips16;\n\t" \
	".type _mcount,@function;\n\t" \
	".ent _mcount\n\t" \
        "_mcount:\n\t" \
        ".frame $sp,88,$31\n\t" \
        ".set noreorder;\n\t" \
        ".set noat;\n\t" \
        PTR_SUBU_STRING " $29,$29,96;\n\t" \
        CPSETUP \
        "sd $4,24($29);\n\t" \
        "sd $5,32($29);\n\t" \
        "sd $6,40($29);\n\t" \
        "sd $7,48($29);\n\t" \
        "sd $8,56($29);\n\t" \
        "sd $9,64($29);\n\t" \
        "sd $10,72($29);\n\t" \
        "sd $11,80($29);\n\t" \
        "sd $2,16($29);\n\t" \
        "sd $1,0($29);\n\t" \
        "sd $31,8($29);\n\t" \
        "move $5,$31;\n\t" \
        "move $4,$1;\n\t" \
        "jal __mcount;\n\t" \
        "nop;\n\t" \
        "ld $4,24($29);\n\t" \
        "ld $5,32($29);\n\t" \
        "ld $6,40($29);\n\t" \
        "ld $7,48($29);\n\t" \
        "ld $8,56($29);\n\t" \
        "ld $9,64($29);\n\t" \
        "ld $10,72($29);\n\t" \
        "ld $11,80($29);\n\t" \
        "ld $2,16($29);\n\t" \
        "ld $31,8($29);\n\t" \
        "ld $1,0($29);\n\t" \
        CPRETURN \
        PTR_ADDU_STRING " $29,$29,96;\n\t" \
        "j $31;\n\t" \
        "move $31,$1;\n\t" \
	".end _mcount;\n\t" \
	".set pop");

#endif
