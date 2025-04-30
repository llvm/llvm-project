/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef GLOBAL_H_
#define GLOBAL_H_

/**
   \file
   \brief FTN global variables and flags.
 */

#include "universal.h"
#include <stdio.h>

#define ARGS_NUMBER 2
#define VECTLEN1 1
#define NUMI_SIZE 4
#define NUMU_SIZE 4
#define POW0 0
#define POW1 1
#define POW2 2

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

#ifdef UTILSYMTAB
typedef int RUTYPE;
#endif

typedef struct {
  int maxsev;      /* maximum error severity for this compile */
  int lineno;      /* current source line number */
  int findex;      /* current file index */
  const char *src_file;  /* name of main input source file */
  const char *curr_file; /* name of current input source file */
  char *module;    /* object module name */
  FILE *srcfil;    /* file pointer for source input file */
  FILE *cppfil;    /* file pointer for preprocessor output */
  FILE *dbgfil;    /* file pointer for debug file */
  FILE *ilmfil;    /* file pointer for (temporary) ILM file */
  FILE *objfil;    /* file pointer for output object file */
  FILE *asmfil;    /* file pointer for output assembly file */
  FILE *stbfil;    /* file pointer for symbols and datatype for llvm compiler */
  int eof_flag;
  const char *ompaccfilename;	/** pointer to the device file name for openmp gpu offload */
  FILE *ompaccfile;	/** file pointer for device code */
  SPTR ompoutlinedfunc;
  SPTR currsub;    /* symtab ptr to current subprogram */
  SPTR caller;     /* symtab ptr to current caller (for bottom-up inlining) */
  int cgr_index;   /* call graph index to current subprogram */
  bool arets;      ///< set to true if any entry contains an alternate return
  RUTYPE rutype;   /* RU_PROG, RU_SUBR, RU_FUNC, or RU_BDATA */
  int funcline;    /* line number of header statement */
  SPTR cmblks;     ///< pointer to list of common blocks
  SPTR externs;    ///< pointer to list of external functions
  SPTR consts;     ///< pointer to list of referenced constants
  SPTR entries;  ///< list of entry symbols
  SPTR statics;   ///< list of "static" variables
  SPTR bssvars;   ///< list of uninitialized "static" variables
  SPTR locals;    ///< pointer to list of local variables
  SPTR basevars; ///< pointer to list of base symbols used for global offsets
  SPTR asgnlbls; ///< pointer to list of labels appearing in assign stmts
  int vfrets;    /* nonzero if variable format (<>) items present */
  ISZ_T caddr;   /* current available address in code space */
  ISZ_T locaddr; /* current available address for local variables,
                  * (positive offset from $local)  */
  ISZ_T saddr;   /* current available address for static variables,
                  * (positive offsets from $static.  */
  ISZ_T
  bss_addr;    /* current available address for static uninitialized variables,
                * (positive offsets from .BSS)  */
  ISZ_T paddr; /* current available address for private variables */
  int prvt_sym_sz;  /* symbol representing size of private area */
  int stk_sym_sz;   /* symbol representing size of stack area */
  int autobj;       /* list of automatic data objects; the st field
                     * AUTOBJ is used to link together the objects; NOSYM
                     * terminates the list.
                     */
  INT silibcnt;     /* number of scheduled ILI blocks */
  char *loc_arasgn; /* pointer to list of ARASGN's for local cmnblk */
  char datetime[21];
  int entbih;           /* entry bih of a function, set by expander/optimizer
                         * to communicate with other modules.
                         */
  int func_count;       /* function counter, current # of function being
                         * compiled, incremented by assem_init */
  const char *file_name; /* full pathname of input file; -file may override */
  int ftn_true;         /* value of .TRUE.; -1 (default) or 1 (-x 125 8) */
  bool has_program;  /* true if a fortran 'program' has been seen */
  bool in_include;   /* set to true if source is from an include file */
  bool nowarn;       /* if TRUE, don't issue warning & informational errors*/
  int internal;         /* internal subprogram state:
                         * 0 - current subprogram does not contain internal
                         *     subprograms.
                         * 1 - current subprogram contains internal subprograms
                         *     (current subprogram is the 'host' subprogram).
                         * >1 - current subprogram is an internal subprogram.
                         */
  SPTR outersub;        /* symtab ptr to containing subprogram */
  SPTR threadprivate;   /* pointer to list of symbols created for each thread-
                         * private common block.  Each symbol will represent
                         * a vector of pointers used to locate a thread's
                         * copy of the common block.
                         */
  bool nofperror;    /* if TRUE, error.c:fperror() does not report errors */
  int fperror_status;   /* error status of a floating point operation
                         * performed by scutil.
                         */
  FILE *ipafil;         /* propagated ipa information */
  FILE *ipofil;         /* newly generated ipa information */
  FILE *dependfil;      /* make dependency information */
  int multiversion;     /* if we're compiling multiple versions of a subprogram
                         */
  int numversions;      /* if we're compiling multiple versions of a subprogram
                         */
  int numcontained;     /* after compiling a host subprogram, how many
                         * contained subprograms are there left to compile */
  int multi_func_count; /* used when compiling multiple versions */
  int pgfi_avail;
  int ec_avail; /* Profile edge count info is available */
  const char *fn; /* name of file being compiled passed from the FE */
  int cuda_constructor;
  int cudaemu; /* emulating CUDA device code */
  int pcast;      /* bitmask for PCAST features */
#ifdef PGF90
  SPTR typedescs; /* list of type descriptors */
#endif
  bool denorm; /* enforce denorm for the current subprogram */
  int outlined;   /* is outlined function .*/
  int usekmpc;    /* use KMPC runtime. turned on for -ta=multicore for llvm. */
#if defined(OMP_OFFLOAD_PGI) || defined(OMP_OFFLOAD_LLVM)
  bool ompaccel_intarget;  /* set when expander is in the openmp target construct */
  bool ompaccel_isdevice;  /* set when generating code for openmp target device */
  SPTR teamPrivateArgs;    /* keeps sptr that holds team private array */
#endif
} GBL;

#undef MAXCPUS
#define MAXCPUS 256

/* mask values for gbl.pcast */
#define PCAST_CODE 1

extern GBL gbl;
#define GBL_CURRFUNC gbl.currsub
#define TPNVERSION 25

typedef struct {
  bool asmcode;
  bool list;
  bool object;
  bool xref;
  bool code;
  bool include;
  bool debug;
  int opt;
  bool depchk;
  bool depwarn;
  bool dclchk;
  bool locchk;
  bool onetrip;
  bool save;
  int inform;
  UINT xoff;
  UINT xon;
  bool ucase;
  char **idir;
  char **linker_directives;
  const char *llvm_target_triple;
  const char *target_features;
  int vscale_range_min;
  int vscale_range_max;
  bool dlines;
  int extend_source;
  bool i4;
  bool line;
  bool symbol;
  int profile;
  bool standard;
  int dbg[96];
  bool dalign; /* TRUE if doubles are double word aligned */
  int astype;     /* target dependent value to support multiple asm's */
  bool recursive;
  int ieee;
  int inliner;
  int autoinline;
  int vect;
  int endian;
  int terse;
  int dollar;   /* defines the char to which '$' is translated */
  int x[252];   /* x flags */
  bool quad; /* quad align "unconstrained objects" if sizeof >= 16 */
  int anno;
  bool qa; /* TRUE => -qa appeared on command line */
  bool es;
  bool p;
  char **def;
  const char *stdinc; /* NULL => use std include; 1 ==> do not look in
                       * std dir; o.w., use value as the std dir */
  bool smp;  /* TRUE => allow smp directives */
  LOGICAL omptarget;  /* TRUE => allow OpenMP Offload directives */
  int errorlimit;
  bool trans_inv; /* global equiv to -Mx,7,0x10000 */
  int tpcount;
  int tpvalue[TPNVERSION]; /* target processor(s), for unified binary */
  const char *cmdline; /* contains compiler command line */
  bool qp; /* Enable quad-precision REAL and quad-precision COMPLEX. */
} FLG;

extern FLG flg;

#define IEEE_CMP (flg.ieee || !XBIT(15, 0x8000000))

#endif // GLOBAL_H_
