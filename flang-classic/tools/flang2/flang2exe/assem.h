/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Assembler module
 *
 * Define those functions and data items which are defined in the assembler
 * module, assem.c, and exported to other modules of the compiler.  (X86 and
 * Hammer targets)
*/

/* ---------- Functions and data used by the Code Generator: ------ */

extern char *comment_char;

char *getsname(SPTR);
char *getsname2(int);
void sym_is_refd(SPTR);

extern void assem_init(void);
/*  called when processing of user function is begun by the CG */

extern void assem_begin_func(int func_sptr);
/*  called when ready to begin emitting code for the function, also
    at each Fortran ENTRY. */

extern void assem_pc(int func_sptr, int arg1_offset, int base_reg);
/*  called after entry code generated - generates precision control
    instructions, if needed.
    arg1_offset - offset from base_reg to arguments argc and argv.
    base_reg - stack pointer or frame pointer register.
    The values of the last two arguments are only used when
    PGC && I386 && WINNT  */

extern void assem_mxcsr(int func_sptr);
/*  called immediately after assem_pc(), to generate SIMD
    control instructions.  */

extern void assem_user_entry(int label_sptr, int lineno, int firstline);
/*  called after all entry code has been emitted (when an ENLAB block
    is encountered).  Indicates the end of the entry code and the
    point where user debugger breakpoints are set.  */

extern void assem_emit_line(int findex, int lineno);
extern void assem_emit_file_line(int findex, int lineno);
/*  called just before emitting code for each basic block  */

extern void assem_end_func(int func_sptr);
/*  called after emitting all code, including exit code  */

extern void assem_data(void);
/*  called after assem_end_func()  */

extern void assem_end(void);
/*  called after assem_data()  */

/* profiling support; these 3 functions are called IFF flg.profile: */

void prof_rouent(int func_sptr);
/*  called immediately after assem_mxcsr() (after emitting the
    entry code for a function)      */

void prof_linent(int lineno, int loop_flag, int blocknum);
/*  called after assem_emit_line(), before emitting the machine
    instructions for a block.  Called only if line profiling is
    requested (flg.profile == 2) and unix-style profiling is not
    requested (xbit(119, 2)).
    Not called if this is an entry block.
    loop_flag - TRUE if this block is the head of a loop.
    blocknum  - bih number of this block    */

void prof_rouret(void);
/*  called just before emitting the exit code for a function.
    Not called if unix-style profiling is requested.        */

extern int get_private_size(void);

extern void add_init_routine(char *initroutine);
/* Create a .init section to call an initialization function */

void create_static_base(int name);

#define STR_SEC 0
#define RO_SEC 1
#define DATA_SEC 2
#define TEXT_SEC 3
#define INIT_SEC 4
#define DRECTVE_SEC 5
#define PDATA_SEC 6
#define XDATA_SEC 7
#define TRACE_SEC 8
#define BSS_SEC 9
#define GXX_EH_SEC 10
/* gnu linkonce.t : weak text */
/* LNK_T_SEC : gcc version 3.99 and below */
/* LNK_T_SEC4 : gcc version 4.0  and above */
#define LNK_T_SEC 11
/* gnu linkonce.d weak comdata data*/
#define LNK_D_SEC 12
/* gnu linkonce.r  weak readonly text for jump tables*/
#define LNK_R_SEC 13
/* gnu older version use ".section .rodata" for jump tables */
/* c++ IA64_ABI  weak rodata for virtual tables */
#define RO_DATA_WEAK_SEC 14
/* c++ IA64_ABI  weak bss for static init variables in template classes */
#define DATA_WEAK_SEC 15
#define BSS_WEAK_SEC 16
#define RO_DATA_SEC 17
#define NVIDIA_FATBIN_SEC 18
#define NVIDIA_MODULEID_SEC 19
#define NVIDIA_RELFATBIN_SEC 20
#define NVIDIA_OLDFATBIN_SEC 21
#define GNU_NAMED_SEC 22
#define OMP_OFFLOAD_SEC 23

#define HAS_TLS_SECTIONS 1

#if defined(HAS_TLS_SECTIONS)
/* .tdata or .tbss for thread local storage */
#define TDATA_SEC 23
#define TBSS_SEC 24
#define TDATA_WEAK_SEC 25
#define TBSS_WEAK_SEC 26
#define DATA_SECG(sptr) (IS_TLS(sptr) ? TDATA_SEC : DATA_SEC)
#define BSS_SECG(sptr) (IS_TLS(sptr) ? TBSS_SEC : BSS_SEC)
#else
#define TDATA_SEC DATA_SEC
#define TBSS_SEC BSS_SEC
#define TDATA_WEAK_SEC DATA_WEAK_SEC
#define TBSS_WEAK_SEC BSS_WEAK_SEC
#define DATA_SECG(sptr) DATA_SEC
#define BSS_SECG(sptr) BSS_SEC
#endif

#if !defined(TARGET_OSX)
#define ULABPFX ".L"
#else
#define ULABPFX "L."
#endif

extern char *comment_char;
extern char *immed_char;
extern char *labpfx;

#define EMIT_INSTR_EXT(op, ext, op1, op2) emit_instr(#op #ext, op1, op2)

/* Conceptually part of assem.c, these filter functions are defined in
   cgassem.c because C and Fortran have separate assem.c files.         */

extern void emit_instr(const char *, const char *, const char *);
/*	emit instruction to assembly file */

extern void emit_label(char *);
/*	output a label to assembly file */

extern char *hex(int num, int pad);
/*  returns a string with num formatted as a hex number padded
    by pad (set pad to 0 if no padding is required) */

extern char *imm(int num);
/*  returns a string with num formatted as an immediate */

extern char *imm_isz(ISZ_T);
/*  returns a string with num, possibly > MAX_INT, formatted as an
    immediate */

extern char *imm_hex(int num, int pad);
/*  returns a string with num formatted as an immediate hex
    number padded by pad zeroes   */

extern char *mem(char *r);
/*  returns a string with r formatted for memory indirection */

extern char *sib(char *base, char *idx, int ss);
/*  returns a string in the appropriate scale-index-base (SIB)
    format */

extern void create_static_name(char *name, int usestatic, int num);

#define EMIT_VZEROUPPER                                             \
  {                                                                 \
    if (!XBIT(164, 0x100000) &&                                     \
        (XBIT(164, 0x2000000) || !TEST_FEATURE(FEATURE_SIMD128)) && \
        mach.feature[FEATURE_AVX])                                  \
      emit_instr("vzeroupper", NULL, NULL);                         \
  }

#define ASM_VZEROUPPER                                              \
  {                                                                 \
    if (!XBIT(164, 0x100000) &&                                     \
        (XBIT(164, 0x2000000) || !TEST_FEATURE(FEATURE_SIMD128)) && \
        mach.feature[FEATURE_AVX])                                  \
      fprintf(ASMFIL, "\tvzeroupper\n");                            \
  }
