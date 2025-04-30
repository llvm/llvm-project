/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Definitions for stab format.
 */

#ifndef _stab_h
#define _stab_h

/*
 * for symbolic debugger, sdb(1):
 */
#define N_GSYM 0x20  /* global symbol: name,,0,type,0 */
#define N_FNAME 0x22 /* procedure name (f77 kludge): name,,0 */
#define N_FUN 0x24   /* procedure: name,,0,linenumber,address */
#define N_STSYM 0x26 /* static symbol: name,,0,type,address */
#define N_LCSYM 0x28 /* .lcomm symbol: name,,0,type,address */
#define N_MAIN 0x2a  /* name of main routine : name,,0,0,0 */
#define N_OBJ 0x38   /* object file path or name */
#define N_OPT 0x3c   /* compiler options */
#define N_RSYM 0x40  /* register sym: name,,0,type,register */
#define N_SLINE 0x44 /* src line: 0,,0,linenumber,address */
#define N_SSYM 0x60  /* structure elt: name,,0,type,struct_offset */
#define N_ENDM 0x62  /* last stab emitted for module */
#define N_SO 0x64    /* source file name: name,,0,0,address */
#define N_LSYM 0x80  /* local sym: name,,0,type,offset */
#define N_BINCL 0x82 /* header file: name,,0,0,0 */
#define N_SOL 0x84   /* #included file name: name,,0,0,address */
#define N_PSYM 0xa0  /* parameter: name,,0,type,offset */
#define N_EINCL 0xa2 /* end of include file */
#define N_ENTRY 0xa4 /* alternate entry: name,linenumber,address */
#define N_LBRAC 0xc0 /* left bracket: 0,,0,nesting level,address */
#define N_EXCL 0xc2  /* excluded include file */
#define N_RBRAC 0xe0 /* right bracket: 0,,0,nesting level,address */
#define N_BCOMM 0xe2 /* begin common: name,, */
#define N_ECOMM 0xe4 /* end common: name,, */
#define N_ECOML 0xe8 /* end common (local name): ,,address */
#define N_LENG 0xfe  /* second stab entry with length information */

/*
 * for the berkeley pascal compiler, pc(1):
 */
#define N_PC 0x30 /* global pascal symbol: name,,0,subtype,line */

/*
 * for modula-2 compiler only
 */
#define N_M2C 0x42   /* compilation unit stab */
#define N_SCOPE 0xc4 /* scope information */

/*
 * for code browser only
 */
#define N_BROWS 0x48 /* path to associated .cb file */

/*
 *    Optional language designations for N_SO
 */

#define N_SO_C 2       /* C          */
#define N_SO_ANSI_C 3  /* Ansi C     */
#define N_SO_CC 4      /* C++	      */
#define N_SO_FORTRAN 5 /* Fortran 77 */

#endif /*!_stab_h*/

/* definitions for coff object file format */
/*
 *   STORAGE CLASSES
 */

#define C_EFCN -1 /* physical end of function */
#define C_NULL 0
#define C_AUTO 1     /* automatic variable */
#define C_EXT 2      /* external symbol */
#define C_STAT 3     /* static */
#define C_REG 4      /* register variable */
#define C_EXTDEF 5   /* external definition */
#define C_LABEL 6    /* label */
#define C_ULABEL 7   /* undefined label */
#define C_MOS 8      /* member of structure */
#define C_ARG 9      /* function argument */
#define C_STRTAG 10  /* structure tag */
#define C_MOU 11     /* member of union */
#define C_UNTAG 12   /* union tag */
#define C_TPDEF 13   /* type definition */
#define C_USTATIC 14 /* undefined static */
#define C_ENTAG 15   /* enumeration tag */
#define C_MOE 16     /* member of enumeration */
#define C_REGPARM 17 /* register parameter */
#define C_FIELD 18   /* bit field */
#define C_BLOCK 100  /* ".bb" or ".eb" */
#define C_FCN 101    /* ".bf" or ".ef" */
#define C_EOS 102    /* end of structure */
#define C_FILE 103   /* file name */

/*		Number of array dimensions in auxiliary entry */
#define DIMNUM 4

/*
   The fundamental type of a symbol packed into the low
   4 bits of the word.
*/

#define T_NULL 0
#define T_ARG 1     /* function argument (only used by compiler) */
#define T_CHAR 2    /* character */
#define T_SHORT 3   /* short integer */
#define T_INT 4     /* integer */
#define T_LONG 5    /* long integer */
#define T_FLOAT 6   /* floating point */
#define T_DOUBLE 7  /* double word */
#define T_STRUCT 8  /* structure  */
#define T_UNION 9   /* union  */
#define T_ENUM 10   /* enumeration  */
#define T_MOE 11    /* member of enumeration */
#define T_UCHAR 12  /* unsigned character */
#define T_USHORT 13 /* unsigned short */
#define T_UINT 14   /* unsigned integer */
#define T_ULONG 15  /* unsigned long */

/*
 * derived types are:
 */

#define DT_NON 0 /* no derived type */
#define DT_PTR 1 /* pointer */
#define DT_FCN 2 /* function */
#define DT_ARY 3 /* array */

/*
 *   type packing constants
 */

#define N_BTMASK 017
#define N_TMASK 060
#define N_TMASK1 0300
#define N_TMASK2 0360
#define N_BTSHFT 4
#define N_TSHIFT 2
