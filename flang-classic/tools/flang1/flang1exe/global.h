/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef GLOBAL_H_
#define GLOBAL_H_

/** \file global.h
    \brief Fortran global variables and flags.
*/


/* this number is - 2^112  */
#define MMAX_MANTI_BIT0_31 0xc06f0000
#define MMAX_MANTI_BIT32_63 0x00000000
#define MMAX_MANTI_BIT64_95 0x00000000
#define MMAX_MANTI_BIT96_127 0x00000000
/* this number is + 2^112  */
#define MAX_MANTI_BIT0_31 0x406f0000
#define MAX_MANTI_BIT32_63 0x00000000
#define MAX_MANTI_BIT64_95 0x00000000
#define MAX_MANTI_BIT96_127 0x00000000

/* this number is + minimum of quad precision */
#define MIN_QUAD_VALUE_BIT96_127 0x00000000

#define MAX_EXP_QVALUE 4931 /* max value exponent */
#define MAX_EXP_OF_QMANTISSA 33 /* 2^112 : 5.19e+33 */
#define EPSILON_BIT96_127 0 /* quad precision allowed error 1.0_16  */
#define REAL_16 16
#define REAL_0 0

#define QVM4_SIZE 4 /* qvm4[] has 4 elements and 128 bits */

/* An index into the symbol table. */
typedef enum SPTR {
  NME_NULL = -1,
  SPTR_NULL = 0,
  SPTR_MAX = 67108864 /* Maximum allowed value */
} SPTR;

#ifdef __cplusplus
// Enable symbol table traversals to work.
static inline void operator++(SPTR &s)
{
  s = SPTR(s + 1);
}
#endif

typedef enum {
  RU_SUBR = 1,
  RU_FUNC,
  RU_PROC,
  RU_PROG,
  RU_BDATA,
} RU_TYPE;

typedef struct {
  int maxsev;      /* maximum error severity for this compile */
  int lineno;      /* current source line number */
  int findex;      /* current file index */
  const char *src_file;  /* name of main input source file */
  const char *curr_file; /* name of current input source file */
  char *module;    /* object module name */
  char *ipaname;   /* IPA database name */
  FILE *srcfil;    /* file pointer for source input file */
  FILE *cppfil;    /* file pointer for preprocessor output */
  FILE *dbgfil;    /* file pointer for debug file */
  FILE *ilmfil;    /* file pointer for (temporary) ILM file */
  FILE *objfil;    /* file pointer for output object file */
  FILE *asmfil;    /* file pointer for output assembly file */
  FILE *outfil;    /* file pointer for source output file */
  FILE *symfil;    /* file pointer for symbol output file */
  FILE *gblfil;    /* file pointer for static global info output file */
  FILE *stbfil;    /* file pointer for symbols and datatypes */
  LOGICAL eof_flag;
  SPTR currsub;     /* symtab ptr to current subprogram */
  SPTR outersub;    /* symtab ptr to host subprogram, if any, or zero */
  int outerentries; /* list of entry symbols to host subprogram, if any, or zero
                       */
  SPTR currmod;     /* symtab ptr to module symbol, if any, or zero */
  LOGICAL arets;    /* set to true if any entry contains an
                       alternate return.  */
  RU_TYPE rutype;   /* RU_PROG, RU_SUBR, RU_FUNC, or RU_BDATA */
  int funcline;     /* line number of header statement */
  int cmblks;       /* pointer to list of common blocks */
  int externs;      /* pointer to list of external functions */
  int consts;       /* pointer to list of referenced constants */
  int entries;      /* list of entry symbols */
  int statics;      /* list of "static" variables */
  int locals;       /* pointer to list of local variables   */
  int asgnlbls;     /* pointer to list of labels appearing in assign stmts.*/
  int ent_select;   /* sptr of int variable whose value (0 .. #entries-1)
                     * denotes which entry was entered. this is zero if
                     * ENTRYs aren't present.
                     */
  int stfuncs;      /* list of statement functions defined in subprogram */
  ISZ_T locaddr;    /* current available address for local variables,
                     * (positive offset from $local)  */
  ISZ_T saddr;      /* current available address for static variables,
                     * (positive offsets from $static.  */
  int autobj;       /* list of automatic data objects; the st field
                     * AUTOBJ is used to link together the objects; NOSYM
                     * terminates the list.
                     */
  int exitstd;      /* pointer to std after which exit code (epilogue)
                     * is added for the current subprogram
                     */
  char datetime[21];
  int entbih;           /* entry bih of a function, set by expander/optimizer
                         * to communicate with other modules.
                         */
  int func_count;       /* function counter, current # of function being
                         * compiled, incremented by assem_init */
  const char *file_name; /* full pathname of input file; -file may override */
  int ftn_true;         /* value of .TRUE.; -1 (default) or 1 (-x 125 8) */
  LOGICAL in_include;   /* set to true if source is from an include file */
  int tp_adjarr;        /* list of template and processor adjustable array
                         * objects; the AUTOBJ st field is the link field;
                         * NOSYM terminates the list.
                         */
  int p_adjarr;         /* pointer to list of based adjustable array-objects;
                         * the SYMLK st field is the link field; NOSYM
                         * terminates the list.
                         */
  int p_adjstr;         /* pointer to list of adjustable lenght string objects;
                         * the SYMLK st field is the link field; NOSYM
                         * terminates the list.
                         */
  LOGICAL nowarn;       /* if TRUE, don't issue warning & informational errors*/
  char *prog_file_name; /* file name containing the 'module', 'program',
                         * 'subroutine', 'function', or 'blockdata' stmt.
                         * follows include files, if necessary. */
  int internal;         /* internal subprogram state:
                         * 0 - current subprogram does not contain internal
                         *     subprograms.
                         * 1 - current subprogram contains internal subprograms
                         *     (current subprogram is the 'host' subprogram).
                         * >1 - current subprogram is an internal subprogram.
                         */
  LOGICAL nofperror;    /* if TRUE, error.c:fperror() does not report errors */
  int fperror_status;   /* error status of a floating point operation
                         * performed by scutil.
                         */
  int sym_nproc;        /* symbol number of hpf$np symbol */
  LOGICAL is_f90;       /* frontend is for Fortran 90 */
  FILE *ipafil;         /* propagated ipa information */
  FILE *ipofil;         /* newly generated ipa information */
  FILE *dependfil;      /* make dependency information */
  FILE *moddependfil;   /* make dependency information */
  const char *fn;       /* name of file being compiled which was previously
                         * preprocessed (can be more general if we choose).
                         */
  LOGICAL denorm;       /* enforce denorm for the current subprogram */
  LOGICAL inomptarget;  /* set if it is OpenMP's target region*/
  LOGICAL empty_contains; /* if TRUE, CONTAINS clause has an empty body */
} GBL;

#undef MAXCPUS
#define MAXCPUS 256

/* Max number of dimensions.  F'2008 requires 15,  Intel is 31. */
#define MAXRANK 7

extern GBL gbl;
#define GBL_CURRFUNC gbl.currsub
#define TPNVERSION 25

typedef struct {
  LOGICAL asmcode;
  LOGICAL list;
  LOGICAL object;
  LOGICAL xref;
  LOGICAL code;
  LOGICAL include;
  LOGICAL output;
  int debug;
  int opt;
  LOGICAL depchk;
  LOGICAL depwarn;
  LOGICAL dclchk;
  LOGICAL locchk;
  LOGICAL onetrip;
  LOGICAL save;
  int inform;
  UINT xoff;
  UINT xon;
  LOGICAL ucase;
  char **idir;
  LOGICAL dlines;
  int extend_source;
  LOGICAL i4;
  LOGICAL line;
  LOGICAL symbol;
  int profile;
  LOGICAL standard;
  int dbg[67];
  LOGICAL dalign; /* TRUE if doubles are double word aligned */
  int astype;     /* target dependent value to support multiple asm's */
  LOGICAL recursive;
  int ieee;
  int inliner;
  int vect;
  LOGICAL endian;
  LOGICAL terse;
  int dollar;   /* defines the char to which '$' is translated */
  int x[252];   /* x flags */
  LOGICAL quad; /* quad align "unconstrained objects" if sizeof >= 16 */
  int anno;
  LOGICAL qa; /* TRUE => -qa appeared on command line */
  LOGICAL es;
  LOGICAL p;
  char **def;
  char **undef;
  const char *stdinc; /* NULL => use std include; 1 ==> do not look in
                       * std dir; o.w., use value as the std dir */
  LOGICAL hpf;
  LOGICAL freeform;
  LOGICAL sequence;
  int ipa;
  LOGICAL craft_supported;
  LOGICAL doprelink; /* generate the .prelink.f file */
  LOGICAL genilm;    /* generate ilm, not fortran, output */
  LOGICAL defaulthpf;
  LOGICAL defaultsequence;
  int errorlimit;
  LOGICAL smp; /* TRUE => allow smp directives */
  LOGICAL omptarget;  /* TRUE => allow OpenMP Offload directives */
  int tpcount;
  int tpvalue[TPNVERSION]; /* target processor(s), for unified binary */
  int accmp;
  const char *cmdline; /* command line used to invoke the compiler */
  LOGICAL qp; /* Enable quad-precision REAL and quad-precision COMPLEX. */
  bool list_macros ; /* TRUE => flang -dM option was used to display macros */
} FLG;

extern FLG flg;

#endif
