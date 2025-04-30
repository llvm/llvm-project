/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file lower.h
    \brief definitions for Fortran front-end's lower module
*/

/*
 * Compatibility History:
 * before 6.2  -- 1.9
 * 6.2         -- 1.10
 *                Includes all of 1.9 + PASSBYVAL & PASSBYREF
 * 7.0         -- 1.11
 *                Includes all of 1.10 + CFUNC for variables
 * 7.1         -- 1.12
 *                Includes all of 1.11 + DECORATE
 * 7.2         -- 1.13
 *                Includes all of 1.12 + CREF & NOMIXEDSTRLEN
 * 8.0         -- 1.14
 *                Includes all of 1.13 + FILE INDEX ENTRIES
 * 8.1         -- 1.15
 *                Includes all of 1.14 + new types + cuda flags
 * 9.0-3       -- 1.16
 *                Includes all of 1.15 + cudaemu value
 * 10.6        -- 1.17
 *                Includes all of 1.16 + sptr for Constant ID data init + denorm
 * 10.9        -- 1.18
 *                Includes all of 1.17 + reflected/mirrored/devcopy flags and
 * devcopy field
 * 11.0        -- 1.19
 *                Includes all of 1.18 + mscall & cref for vars & members
 * 11.4        -- 1.20
 *                Includes all of 1.18 + libm & libc for functions
 * 12.7        -- 1.21
 *                Includes all of 1.20 + TASK for variables
 * 12.7        -- 1.22
 *                Includes all of 1.21 + cuda texture flag
 * 12.7        -- 1.23
 *                Includes all of 1.22 + INTENTIN flag
 * 13.0        -- 1.24
 *                Includes all of 1.23 + DATACNST flag
 * 13.5        -- 1.25
 *                Includes all of 1.24 + MODCMN flag
 * 13.8        -- 1.26
 *                Includes all of 1.25 + DOBEGNZ & DOENDNZ
 * 13.9        -- 1.27
 *                Includes all of 1.26 + symbol ACCCREATE and ACCRESIDENT
 * 14.0        -- 1.28
 *                Includes all of 1.27 + ACCROUT
 * 14.4        -- 1.29
 *                Includes all of 1.28 + CUDAMODULE
 * 14.4        -- 1.30
 *                Includes all of 1.29 + MANAGED + additionsl ILM operand
 *                    for the call ILMs via a procedure ptr, e.g., CALLA,
 *                    CDFUNCA, etc.
 * 14.7        -- 1.31
 *                All of 1.30 + ACCCREATE/ACCRESIDENT for common blocks,
 *                    +ACCLINK+ACCCOPYIN flags
 * 15.0        -- 1.32
 *                All of 1.31 + new FARGF ILM
 * 15.3        -- 1.33
 *                All of 1.32 + FWDREF flag + INTERNREF flag + AGOTO field
 * 15.4        -- 1.34
 *                All of 1.33 + ST_MEMBER IFACE field
 * 15.7        -- 1.35
 *                All of 1.34 + ST_ENTRY/ST_PROC ARET field
 * 15.9        -- 1.36
 *                All of 1.35 + PARREF, PARSYMS, and PARSYMSCT field
 * 15.10       -- 1.37
 *                All of 1.36 + IM_BMPSCOPE/IM_EMPSCOPE
 * 16.0        -- 1.38
 *                All of 1.37 + IM_MPSCHED/IM_MPLOOP and
 *                   IM_MMBORDERED/IM_MPEORDERED + TPALLOC + IM_FLUSH flag
 * 16.4        -- 1.39
 *                All of 1.38 + IM_ETASK and IM_TASKFIRSPRIV
 * 16.5        -- 1.40
 *                All of 1.39 + ISOTYPE flag + update IM_MPSCHED and IM_MPLOOP
 * 16.6        -- 1.41
 *                All of 1.40 + IM_LSECTION
 * 16.6        -- 1.42
 *                All of 1.41 + VARARG
 * 16.8        -- 1.43
 *                All of 1.42 + ALLOCATTR + F90POINTER
 * 16.10       -- 1.44
 *                All of 1.43 +
 *IM_TASKGROUP/ETASKGROUP/TARGET/TARGETDATA/TARGETUPDATE/
 *                TARGETEXITDATA/TARGETENTERDATA/DISTRIBUTE/TEAMS and their
 *combinations
 *		  TARGET/TEAMS/DISTRIBUTE/PARALLEL DO/CANCEL/CANCELLATIONPOINT
 *                constructs
 * 17.0        -- 1.45
 *                All of 1.44 + INVOBJINC + PARREF for ST_PROC
 * 17.2        -- 1.46
 *                All of 1.45 + etls + tls, irrspective of target
 * 17.7        -- 1.47
 *                All of 1.46 + BPARA + PROC_BIND + MP_ATOMIC...
 * 17.10        -- 1.48
 *                All of 1.47 + ETASKFIRSTPRIV, MP_[E]TASKLOOP,
 *                MP_[E]TASKLOOPREG
 * 18.1         -- 1.49
 *                All of 1.48 , MP_TASKLOOPVARS, [B/E]TASKDUP
 * 18.4
 *              -- 1.50
 *                All of 1.49 +
 *                Internal procedures passed as arguments and pointer targets
 * 18.7         -- 1.51
 *                All of 1.50 +
 *                remove parsyms field and add parent for ST_BLOCK,
 *                pass "has_opts" (no optional arguments) flag for ST_ENTRY and
 *                ST_PROC symbols to back-end.
 * 18.10        -- 1.52
 *                All of 1.51 +
 *                add IS_INTERFACE flag for ST_PROC, and for ST_MODULE when emitting
 *                as ST_PROC
 * 19.3         -- 1.53
 *                All of 1.52 +
 *                Add has_alias bit, and length and name of the alias for Fortran
 *                module variable when it is on the ONLY list of a USE statement.
 *                This is for Fortran LLVM compiler only.
 *
 * 19.10        -- 1.54
 *              All of 1.53 +
 *              pass allocptr and ptrtarget values for default initialization
 *              of standalone pointers.
 *
 * 20.1         -- 1.55
 *              All of 1.54 +
 *              pass elemental field for subprogram when emitting ST_ENTRY.
 *
 *              For ST_PROC, pass IS_PROC_PTR_IFACE flag.
 *
 * 23.12        -- 1.56
 *              All of 1.55 + PALIGN
 */
#define VersionMajor 1
#define VersionMinor 56

void lower(int);
void lower_end_contains(void);
void create_static_base(int blockname);

#ifdef INSIDE_LOWER
#include <stdarg.h>

#if DEBUG
#define Trace(a) LowerTraceOutput a
/* print a message, continue */
void LowerTraceOutput(const char *fmt, ...);
#else
/* eliminate the trace output */
#define Trace(a)
#endif

void lerror(const char *fmt, ...);
void lower_visit_symbol(int sptr);
void lower_finish_sym(void);
void lower_use_datatype(int dtype, int usage);
void lower_data_types(void);
void lower_namelist_plists(void);
void lower_pointer_init(void);
void lower_push(int value);
void lower_check_stack(int);
void lower_linearized(void);
void lower_common_sizes(void);
int lower_pop(void);
int lower_getintcon(int val);
int lower_getiszcon(ISZ_T val);
int lower_getrealcon(int val);
int lower_getlogcon(int val);
int lower_newfunc(const char *name, int stype, int dtype, int sclass);
void lower_add_pghpf_commons(void);
void lower_symbols(void);
void lower_clear_visit_fields(void);
void lower_set_symbols(void);
void lower_init_sym(void);
void lower_fill_member_parent(void);
void lower_sym_header(void);
void lower_fileinfo(void);
void lower_mark_entries(void);
int lower_makefunc(const char *name, int dtype, LOGICAL isDscSafe);
int lower_lab(void);
void lower_check_generics(void);

struct lower_syms {
  int license, localmode, ptr0, ptr0c;
  int intzero, intone, realzero, dblezero, quadzero;
  /* pointers for functions: loc, exit, allocate */
  int loc, exit, alloc, alloc_chk, ptr_alloc, dealloc, dealloc_mbr, lmalloc,
      lfree;
  int calloc, ptr_calloc;
  int auto_alloc, auto_calloc, auto_dealloc;
  int oldsymavl, outersub, outerentries;
  int ptrnull;

  int docount, labelcount, first_outer_sym, last_outer_sym_orig, last_outer_sym,
      acount, Ccount;
  int sym_lineno, last_lineno, sym_line_entry, sym_local, sym_save_local,
      sym_function_entry, sym_function_exit, sym_exit, sym_function_name,
      sym_file_name;
  int sym_subchk, sym_ptrchk, sym_chkfile, intmax;
  int sched_dtype;
  int scheds_dtype;
  int parallel_depth, task_depth, sc;
  FILE *lowerfile;
  /*
   * The following members are initialized to values which reflect the
   * default type for the extents and subscripts of arrays.  The type could
   * either be 32-int or 64-bit (BIGOBJects & -Mlarge_arrays).
   */
  struct {
    int zero;  /* Predefined sym for ISZ_T 0 (stb.i0 or stb.k0) */
    int one;   /* Predefined sym for ISZ_T 1 (stb.i1 or stb.k1) */
    int max;   /* Predefined sym for ISZ_T MAX */
    int dtype; /* Type used for extents and subscripts. */
    /* ilms for subscript operations (e.g., "ILD" or "KLD"): */
    const char *load;
    const char *store;
    const char *con;
    const char *add;
    const char *sub;
    const char *mul;
    const char *div;
  } bnd;
  struct {
    int dtype;
    int kput;
    char *alloc;
    char *calloc;
    char *ptr_alloc;
    char *ptr_calloc;
  } allo;
};
extern struct lower_syms lowersym;

extern struct ref_symbol dbgref_symbol;

typedef struct {
  int member_parent;   /* pointer from each 'member' to the 'parent' structure type symbol */
  int symbol_replace;  /* When one symbol must be replaced by another, set its value here */
  int pointer_list;    /* linked list of pointer or allocatable variables whose
                        * pointer/offset/descriptors need to be initialized */
  int refd_list;       /* linked list of pointer/offset/section descriptors in the order they
                        * need to be given addresses */
} lower_symbol_lists;

struct lsymlists_s {
  STG_MEMBERS(lower_symbol_lists);
};
extern struct lsymlists_s lsymlists;

#define LOWER_MEMBER_PARENT(x) lsymlists.stg_base[x].member_parent
#define LOWER_SYMBOL_REPLACE(x) lsymlists.stg_base[x].symbol_replace
#define LOWER_POINTER_LIST(x) lsymlists.stg_base[x].pointer_list
#define LOWER_REFD_LIST(x) lsymlists.stg_base[x].refd_list

extern int *lower_argument;
extern int lower_argument_size;
extern int lower_line;

/* only one of thenlabel and elselabel should be nonzero;
 * the other is the 'fall through' case */
typedef struct {
  int thenlabel, elselabel, endlabel;
} iflabeltype;

extern int lower_disable_ptr_chk;
extern int lower_disable_subscr_chk;

/* types of entries pushed onto the stack */
#define STKDO 1
#define STKIF 2
#define STKSINGLE 3
#define STKSECTION 4
#define STKMASTER 5
#define STKTASK 6
#define STKCANCEL 7
#define STKDDO 8

void lower_ilm_header(void);
int plower(const char *fmt, ...);
int plower_arg(const char *, int, int, int);
void lower_start_stmt(int lineno, int label, LOGICAL exec, int std);
void lower_end_stmt(int std);
void lower_stmt(int std, int ast, int lineno, int label);
int lower_base(int ast);
int lower_base_sptr(int sptr);
int lower_address(int ast);
int lower_target(int ast);
int lower_ilm(int ast);
void lower_expression(int ast);
void lower_reinit(void);
void lower_exp_finish(void);
void lower_data_stmts(void);
void lower_debug_label(void);
void lower_ilm_finish(void);

/* manage lower-created temporaries */
void lower_reset_temps(void);
int lower_scalar_temp(int);

/* save spaces for ILM and BASE ILM info */
#define A_ILMP(ast, ilm) A_OPT1P(ast, ilm)
#define A_ILMG(ast) A_OPT1G(ast)
#define A_BASEP(ast, ext) A_OPT2P(ast, ext)
#define A_BASEG(ast) A_OPT2G(ast)
#if DEBUG
int lower_ndtypeg(int);
#define NDTYPE_IS_SET(ast) (astb.stg_base[ast].w19 > 0)
#undef A_NDTYPEG
#define A_NDTYPEG(ast) lower_ndtypeg(ast)
#else
#define NDTYPE_IS_SET(ast) (A_NDTYPEG(ast) > 0)
#endif

int lower_conv(int ast, int dtype);
int lower_conv_ilm(int ast, int ilm, int fromdtype, int todtype);
int lower_null(void);
int lower_null_arg(void);
int lower_nullc_arg(void);
void lower_logical(int, iflabeltype *);
char *ltyped(const char *opname, int dtype);
void ast_error(const char *s, int ast);
void lower_clear_opt(int ast, int *unused);
int lower_parenthesize_expression(int ast);
int lower_typestore(int dtype, int lilm, int rilm);
int lower_typeload(int dtype, int ilm);
void lower_check_pointer(int ast, int ilm);
void lower_check_subscript(int sym, int ast, int ndim, int *ilm, int *lower,
                           int *upper);

void ccff_lower(FILE *lfile); /* ccffinfo.c */

void uncouple_callee_args(void); /* lowersym.c */
void lower_unset_symbols(void);
void lower_outer_symbols(void);
void lower_set_craypointer(void);
void stb_fixup_llvmiface(void);

void fill_entry_bounds(int sptr, int lineno); /* lowerilm.c */
int lower_replacement(int ast, int sym);
int lowersym_pghpf_cmem(int *whichmem);

/*
 *  The following are used to determine how to return bind(C) function retvals
 * according to the ABI
 */
/* Classes of by value arguments and C bind retvals */
#define CLASS_NONE 0
#define CLASS_INT1 1
#define CLASS_INT2 2
#define CLASS_INT3 3
#define CLASS_INT4 4
#define CLASS_INT5 5
#define CLASS_INT6 6
#define CLASS_INT7 7
#define CLASS_INT8 8
#define CLASS_SSESP4 9
#define CLASS_SSESP8 10
#define CLASS_SSEDP 11
#define CLASS_SSEQ 12
#define CLASS_MEM 13
#define CLASS_FSTK 14 // TODO: UNUSEDS delete
#define CLASS_PTR 15

/* mostly used for small structs passed in regs stuff.*/
/* These values must be kept insync with the values in the BE file exp_rte.c */
#if defined(TARGET_WIN_X8664)
#define MAX_PASS_STRUCT_SIZE 8
#else
#define MAX_PASS_STRUCT_SIZE 16
#endif

int check_return(int retdtype);

#endif
