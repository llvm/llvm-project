/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef SEMANT_H_
#define SEMANT_H_

/**
   \file
   \brief Fortran semantic analyzer data definitions.
 */

#include "symtab.h"

#define S_NULL 0
#define S_CONST 1
#define S_EXPR 2
#define S_LVALUE 3
#define S_LOGEXPR 4
#define S_STAR 5
#define S_VAL 6
#define S_IDENT 7
#define S_LABEL 8
#define S_STFUNC 9
#define S_REF 10
#define S_TRIPLE 11
#define S_KEYWORD 12

#define OP_NEG 0
#define OP_ADD 1
#define OP_SUB 2
#define OP_MUL 3
#define OP_DIV 4
#define OP_XTOI 5
#define OP_XTOK 6
#define OP_XTOX 7
#define OP_CMP 8
#define OP_AIF 9
#define OP_LD 10
#define OP_ST 11
#define OP_FUNC 12
#define OP_CON 13
#define OP_CAT 14
#define OP_LOG 15
#define OP_LEQV 16
#define OP_LNEQV 17
#define OP_LOR 18
#define OP_LAND 19
#define OP_EQ 20
#define OP_GE 21
#define OP_GT 22
#define OP_LE 23
#define OP_LT 24
#define OP_NE 25
#define OP_LNOT 26

/* Different types of atomic actions. */
#define ATOMIC_UNDEF -1
#define ATOMIC_UPDATE 1
#define ATOMIC_READ 2
#define ATOMIC_WRITE 3
#define ATOMIC_CAPTURE 4

/*
 * Generate lexical block debug information?  Criteria:
 *    -debug
 *    not -Mvect   (!flg.vect)
 *    not -Mconcur (!XBIT(34,0x200)
 *    lex block disabled (!XBIT(123,0x4000))
 */
#define DBG_LEXBLK \
  flg.debug && !flg.vect && !XBIT(34, 0x200) && !XBIT(123, 0x400)

typedef struct xyyz {
  struct xyyz *next;
  union {
    int sptr;
    int ilm;
    struct sst *stkp;
    INT conval;
    ISZ_T szv;
  } t;
} ITEM;
#define ITEM_END ((ITEM *)1)

typedef struct {
  int index_var; /* do index variable */
  int count_var;
  int top_label;
  int zerot_label;
  int init_expr;
  int limit_expr;
  int step_expr;
  int lastval_var;
  int collapse;    /* collapse level if loop is within a collapse set of
                    * loops; 1 is innermost
                    */
  char prev_dovar; /* DOVAR flag of index variable before it's entered */
} DOINFO;

typedef struct reduc_sym {
  int shared;  /* shared symbol */
  int Private; /* private copy */
  struct reduc_sym *next;
} REDUC_SYM;

typedef struct reduc_tag { /* reduction clause item */
  int opr;                 /* if != 0, OP_xxx value */
  int intrin;              /* if != 0, sptr to intrinsic */
  REDUC_SYM *list;         /* list of shared variables & private copies */
  struct reduc_tag *next;
} REDUC;

typedef struct noscope_sym {
  int oldsptr;
  int newsptr;
  int lineno;
  bool is_dovar;
} NOSCOPE_SYM;

typedef struct { /* DO-IF stack entries */
  int Id;
  int lineno;     /* beginning line# of control structure */
  int nest;       /* bit vector indicating the structures are present
                   * in the stack including the current structure
                   */
  int name;       /* index into the symbol names area representing the
                   * name of the construct; 0 if construct is not named.
                   */
  int exit_label; /* For a DO loop, the label (target) of an EXIT
                   * stmt; 0 if the EXIT stmt is not present.
                   * For block-if construct, the label of the
                   * statement after the matching endif; created
                   * if else-if is present; 0 if else-if not present.
                   * For a case-construct, label of the statement after
                   * the case construct
                   */
  union {
    struct { /* IF statements */
      int false_lab;
    } u1;
    struct { /* DO statements */
      int do_label;
      int cycle_label;
      DOINFO *doinfo;
    } u2;
  } u;
  struct {                      /* OpenMP stuff */
    REDUC *reduc;               /* reductions for parallel constructs */
    REDUC_SYM *lastprivate;     /* lastprivate for parallel constructs */
    ITEM *allocated;            /* list of allocated private variables */
    ITEM *dfltal;               /* list of default private/firstprivate
                                 * allocated variables.
                                 */
    REDUC_SYM *dfltfp;          /* list of default firstprivate variables */
    NOSCOPE_SYM *no_scope_base; /* list of variables without scope
                                 * with default(none)
                                 */
    int no_scope_avail;
    int no_scope_size;

    union {
      struct {        /* parallel sections */
        int sect_lab; /* sptr of label beginning a SECTIONS
                       * or SECTION. sptr of the global
                       * semaphore variable created for
                       * CRITICAL <ident>
                       */
        int sect_cnt; /* number of SECTION blocks */
        int sect_var; /* where to store section number */
      } v1;
      struct {           /* parallel do statements */
        int sched_type;  /* one of DI_SCHxxx if a parallel do */
        int chunk;       /* When the parallel do is parsed, this
                          * field is the sptr representing the chunk size
                          * (0 if not present). When the parallel do's
                          * corresponding DO statement is processed, this
                          * field, possibly NULL, is a 'doinfo' record whose
                          * interpretation depends on the scheduling type:
                          *     DI_SCH_STATIC - information for iterating
                          *                     thru a chunk.
                          *     Other         - information for the outer
                          *                     scheduling loop.
                          */
        bool is_ordered; /* loop has the ordered attribute */
      } v2;
    } v;
  } omp;
} DOIF;

#define DI_IF 0
#define DI_DO 1
#define DI_DOW 2
#define DI_PAR 3
#define DI_PARDO 4
#define DI_PDO 5
#define DI_DOACROSS 6
#define DI_PARSECTS 7
#define DI_SECTS 8
#define DI_SINGLE 9
#define DI_CRITICAL 10
#define DI_MASTER 11
#define DI_ORDERED 12
#define DI_TASK 13
#define DI_ATOMIC_CAPTURE 14

#define DI_SCH_STATIC 0
#define DI_SCH_DYNAMIC 1
#define DI_SCH_GUIDED 2
#define DI_SCH_INTERLEAVE 3
#define DI_SCH_RUNTIME 4
#define DI_SCH_AUTO 5
#define DI_SCH_DIST_STATIC 6

#define DI_ID(d) sem.doif_base[d].Id
#define DI_LINENO(d) sem.doif_base[d].lineno
#define DI_NEST(d) sem.doif_base[d].nest
#define DI_NAME(d) sem.doif_base[d].name
#define DI_EXIT_LAB(d) sem.doif_base[d].exit_label
#define DI_FALSE_LAB(d) sem.doif_base[d].u.u1.false_lab
#define DI_DO_LABEL(d) sem.doif_base[d].u.u2.do_label
#define DI_CYCLE_LABEL(d) sem.doif_base[d].u.u2.cycle_label
#define DI_DOINFO(d) sem.doif_base[d].u.u2.doinfo
#define DI_SECT_LAB(d) sem.doif_base[d].omp.v.v1.sect_lab
#define DI_SECT_CNT(d) sem.doif_base[d].omp.v.v1.sect_cnt
#define DI_SECT_VAR(d) sem.doif_base[d].omp.v.v1.sect_var
#define DI_NOSCOPE_BASE(d) sem.doif_base[d].omp.no_scope_base
#define DI_NOSCOPE_SIZE(d) sem.doif_base[d].omp.no_scope_size
#define DI_NOSCOPE_AVL(d) sem.doif_base[d].omp.no_scope_avail
#define DI_CRITSYM(d) sem.doif_base[d].omp.v.v1.sect_lab
#define DI_SCHED_TYPE(d) sem.doif_base[d].omp.v.v2.sched_type
#define DI_CHUNK(d) sem.doif_base[d].omp.v.v2.chunk
#define DI_IS_ORDERED(d) sem.doif_base[d].omp.v.v2.is_ordered
#define DI_REDUC(d) sem.doif_base[d].omp.reduc
#define DI_LASTPRIVATE(d) sem.doif_base[d].omp.lastprivate
#define DI_ALLOCATED(d) sem.doif_base[d].omp.allocated
#define DI_DFLT_ALLOCATED(d) sem.doif_base[d].omp.dfltal
#define DI_DFLT_FIRSTPRIVATE(d) sem.doif_base[d].omp.dfltfp
#define DI_B(t) (1 << t)
#define DI_IN_NEST(d, t) (DI_NEST(d) & DI_B(t))

#define NEED_LOOP(df, typ)                                               \
  {                                                                      \
    df = ++sem.doif_depth;                                               \
    NEED(df + 1, sem.doif_base, DOIF, sem.doif_size, sem.doif_size + 8); \
    DI_EXIT_LAB(df) = DI_CYCLE_LABEL(df) = 0;                            \
    DI_NAME(df) = 0;                                                     \
    DI_LINENO(df) = gbl.lineno;                                          \
    DI_ID(df) = typ;                                                     \
    DI_NOSCOPE_AVL(df) = 0;                                              \
    DI_NOSCOPE_SIZE(df) = 0;                                             \
    DI_NOSCOPE_BASE(df) = NULL;                                          \
    DI_NEST(df) = DI_NEST(df - 1) | DI_B(typ);                           \
  }

/* Define Initializer Variable List */
typedef struct VAR { /* used for elements of dinit variable list */
  short id;
#define Dostart 0
#define Doend 1
#define Varref 2
  union {
    struct {
      int indvar;
      int lowbd, upbd;
      int step;
    } dostart;
    struct {
      struct VAR *dostart;
    } doend;
    struct {
      /* Semantic stack info for variable reference */
      int id;
      int ptr; /* May be symbol ptr or ilm ptr */
      DTYPE dtype;
      int shape;
    } varref;
  } u;
  struct VAR *next;
} VAR;

/* Define Initializer Constant Tree */
typedef struct CONST CONST;

typedef struct {
  SPTR index_var; /* sptr of index variable */
  CONST *initval;
  CONST *limitval;
  CONST *stepval;
} IDOINFO;

typedef struct AEXPR {
  int op;
  CONST *lop;
  CONST *rop;
} AEXPR;

struct CONST {
  char id;
  CONST *next;
  CONST *subc;
  ISZ_T repeatc;
  SPTR sptr;
  SPTR mbr; /* will be the sptr of the member when the initializer is an IDENT
             * (presumbably, a PARAMETER) */
  DTYPE dtype;
  int no_dinitp;
  union {
    INT conval;
    AEXPR expr;
    IDOINFO ido;
  } u1;
};

/***** KEEP AC values consistent with the front-end *****/
#define AC_IDENT 1
#define AC_CONST 2
#define AC_EXPR 3  /* SST expr */
#define AC_IEXPR 4 /* AC expression */
#define AC_AST 5
#define AC_IDO 6
#define AC_REPEAT 7
#define AC_ACONST 8
#define AC_SCONST 9
#define AC_LIST 10 /* only used during DATA stmt processing */
#define AC_VMSSTRUCT 11
#define AC_VMSUNION 12
#define AC_TYPEINIT 13
#define AC_ICONST                                      \
  14 /* integer constant value, currently used to keep \
      * intrinsic routine selector                     \
      */
#define AC_CONVAL                                       \
  15 /* Type of ACL leaf item generated by calling      \
      * eval_init_expr/eval_init_expr_item. The conval  \
      * field contains the results of the evaluation.   \
      * The type of the value is a literal constant if  \
      * the type a TY_WORD. Otherwise, the value is the \
      * sptr of a constant.                             \
      */
#define AC_ADD 1
#define AC_SUB 2
#define AC_MUL 3
#define AC_DIV 4
#define AC_EXP 5
#define AC_NEG 6
#define AC_INTR_CALL 7
#define AC_ARRAYREF 8
#define AC_MEMBR_SEL 9
#define AC_CONV 10
#define AC_CAT 11
#define AC_EXPK 12
#define AC_LEQV 13
#define AC_LNEQV 14
#define AC_LOR 15
#define AC_LAND 16
#define AC_EQ 17
#define AC_GE 18
#define AC_GT 19
#define AC_LE 20
#define AC_LT 21
#define AC_NE 22
#define AC_LNOT 23
#define AC_EXPX 24
#define AC_TRIPLE 25

#define AC_I_adjustl 1
#define AC_I_adjustr 2
#define AC_I_char 3
#define AC_I_ichar 4
#define AC_I_index 5
#define AC_I_int 6
#define AC_I_ishft 7
#define AC_I_ishftc 8
#define AC_I_kind 9
#define AC_I_lbound 10
#define AC_I_len 11
#define AC_I_len_trim 12
#define AC_I_nint 13
#define AC_I_null 14
#define AC_I_repeat 15
#define AC_I_reshape 16
#define AC_I_scan 17
#define AC_I_selected_int_kind 18
#define AC_I_selected_real_kind 19
#define AC_I_size 20
#define AC_I_transfer 21
#define AC_I_trim 22
#define AC_I_ubound 23
#define AC_I_verify 24
#define AC_I_shape 25
#define AC_I_min 26
#define AC_I_max 27
#define AC_I_fltconvert 28
#define AC_I_floor 29
#define AC_I_ceiling 30
#define AC_I_mod 31
#define AC_I_sqrt 32
#define AC_I_exp 33
#define AC_I_log 34
#define AC_I_log10 35
#define AC_I_sin 36
#define AC_I_cos 37
#define AC_I_tan 38
#define AC_I_asin 39
#define AC_I_acos 40
#define AC_I_atan 41
#define AC_I_atan2 42
#define AC_I_selected_char_kind 43
#define AC_I_abs 44
#define AC_I_iand 45
#define AC_I_ior 46
#define AC_I_ieor 47
#define AC_I_merge 48
#define AC_I_lshift 49
#define AC_I_rshift 50
#define AC_I_maxloc 51
#define AC_I_maxval 52
#define AC_I_minloc 53
#define AC_I_minval 54
#define AC_I_scale 55
#define AC_I_transpose 56
#define AC_UNARY_OP(e) (e.op == AC_NEG || e.op == AC_CONV)

typedef struct {  /* STRUCTURE stack entries */
  char type;      /* 's': STRUCTURE; 'u': UNION; 'm: MAP */
  int sptr;       /* Sym ptr to field name list having this structure */
  int dtype;      /* Pointer to structure dtype */
  int last;       /* last member; updated by link_members */
  CONST *ict_beg; /* Initializer Constant Tree begin */
  CONST *ict_end; /* Initializer Constant Tree end */
} STSK;
/* access entries in STRUCTURE stack; 0 ==> top of stack, 1 ==> 1 back, etc. */
#define STSK_ENT(i) sem.stsk_base[sem.stsk_depth - (i)-1]

typedef struct equiv_var { /* variable references in EQUIVALENCE statements */
  int sptr;
  int lineno;
  ITEM *subscripts;
  ISZ_T byte_offset;
  struct equiv_var *next;
  /* the next field can be made smaller if more fields must be added */
  INT is_first; /* first in a group */
} EQVV;
#define EQVV_END ((EQVV *)1)

/*  define structures needed for statement function processing: */

typedef struct _sfuse {
  char usetyp; /* type of use:
                * 0 - value
                * 1 - address (loc intrinsic)
                * 2 - argument to a function
                */
  struct _sfuse *next;
  int ilm;
} SFUSE;

typedef struct arginfo {
  int ilm[3];           /* flags/ilms corresponding to usetyp in SFUSE:
                         * when searching for uses of the formal, marks whether or
                         * not its value ([0]) is needed, it appears in the loc
                         * intrinsic ([1]), or if it appears as an argument to
                         * a function ([2]).
                         * during evaluation, this array will locate the ilms
                         * suitable for use as a value, loc operand, or as an
                         * argument, respectively.
                         */
  int dtype;            /* data type of dummy argument  */
  SFUSE *uses;          /* ptr to list of ITEM records locating the
                         * the uses of this dummy arg within the ILMS
                         */
  struct arginfo *next; /* next argument info record */
} ARGINFO;

typedef struct {    /*  statement function descriptor  */
  ILM_T *ilmp;      /* ptr to ILM's */
  ARGINFO *args;    /* ptr to list of arginfo records */
  SFUSE *links;     /* ptr to list of links to be relocated */
  int rootilm;      /* root of expression tree, 0 if none */
  ARGINFO *ident;   /* for s.f. of form f(a) = a, points to arginfo
                     * record for a */
  SFUSE *new_temps; /* ptr to list of ILMs using temps which were created
                     * when the statement function was defined and need to
                     * be replaced when the statement function is
                     * referenced.
                     */
} SFDSC;

/*
 * define a stack for scope entries -- currently only used when entering
 * parallel regions:
 *   a 'zero' level scope is for the outer/subprogram level.
 *   n > 0 - parallel nesting level.
 * The scope stack is indexed by sem.scope.
 */
typedef struct scope_sym_tag {
  int sptr;  /* symbol appearing in the SHARED clause */
  int scope; /* its outer scope value */
  struct scope_sym_tag *next;
} SCOPE_SYM;

#define PAR_SCOPE_NONE 0
#define PAR_SCOPE_SHARED 1
#define PAR_SCOPE_PRIVATE 2
#define PAR_SCOPE_FIRSTPRIVATE 3
#define PAR_SCOPE_TASKNODEFAULT 4

typedef struct {
  int rgn_scope;          /* index of the scope entry of the containing
                           * parallel region.
                           */
  int par_scope;          /* one of PAR_SCOPE_... */
  int di_par;             /* index of the DOIF structure corresponding to
                           * this scope.
                           */
  int sym;                /* the ST_BLOCK defining this scope */
  int autobj;             /* list of automatic data objects for this
                           * scope
                           */
  int prev_sc;            /* previous storage class */
  SCOPE_SYM *shared_list; /* List of symbols appearing in the SHARED
                           * clause for this scope when par_scope is
                           * 'shared'.
                           */
} SCOPESTACK;

#define BLK_SYM(i) sem.scope_stack[i].sym
#define BLK_AUTOBJ(i) sem.scope_stack[i].autobj

/*  declare global semant variables:  */

typedef struct {
  bool wrilms;        /* set to FALSE if don't need to write ILM's */
  int doif_size;      /* size in records of DOIF stack area.  */
  DOIF *doif_base;    /* base pointer for DOIF stack area. */
  int doif_depth;     /* current DO-IF nesting level */
  EQVV *eqvlist;      /* pointer to head of equivalence list */
  int flabels;        /* pointer to list of ftn ref'd labels */
  SPTR nml;           /* pointer to list of namelist symbols */
  int funcval;        /* pointer to variable for function ret val */
  int pgphase;        /* statement type seen so far:
                       *
                       *  0 - nothing seen yet (initial value)
                       *  1 - SUBROUTINE, FUNCTION, BLOCKDATA,
                       *      PROGRAM
                       *  2 - Specification statements
                       *  3 - DATA statements or statement function
                       *      definitions
                       *  4 - Executable statements
                       *  5 - END statement
                       *
                       *  NOTES:
                       *     PARAMETER, NAMELIST, and IMPLICIT do not
                       *     explicitly set pgphase unless pgphase is
                       *     0 in which case it's set to 1. These are
                       *     allowed between pgphases 0/1 and 2.
                       */
  int gdtype;         /* global data type */
  int ogdtype;        /* original global data type (i.e. before *n
                         modification */
  int gcvlen;         /* global character type size */
  int atemps;         /* avail counter for array bounds temporaries */
  int itemps;         /* avail counter for temporaries named 'ixxx' */
  int ptemps;         /* avail counter for inliner ptr temporaries */
  bool savall;        /* SAVE statement w.o. symbols specified */
  bool savloc;        /* at least one local variable SAVE'd */
  bool none_implicit; /* insure that variables are declared - set
                            TRUE if IMPLICIT NONE seen */
  STSK *stsk_base;    /* base pointer for structure stack area */
  int stsk_size;      /* size in records of structure stack area */
  int stsk_depth;     /* current structure depth (i.e. stack top) */
  int stag_dtype;     /* structure tag dtype pointer */
  int psfunc;         /* next <var ref> may be lhs of statement func */
  int dinit_error;    /* error flag during DATA stmt processing */
  int dinit_count;    /* # elements left in current dcl id to init */
  bool dinit_data;    /* TRUE if in DATA stmt, FALSE if type dcl or
                            structure init stmt */
  struct {            /* info for variable format expression */
    int temps;        /*   counter for temporary labels */
    int labels;       /*   pointer to list of vfe labels */
  } vf_expr;
  bool ignore_stmt;  /* TRUE => parser is to ignore current stmt */
  int switch_size;   /* size of switch/CGOTO list area */
  int switch_avl;    /* next available word in switch list area */
  int bu_switch_avl; /* switch_avl for bottom-up Minline */
  bool temps_reset;  /* TRUE if semant general temps can be resused */
  bool in_stfunc;    /* in statement function def */
  int p_adjarr;      /* pointer to list of based adjustable array-objects */
  int in_dim;        /* in <dimension list> */
                     /*
                      * the following two members (bounds, and arrdim) are filled in
                      * when semantically processing <dim list> specifiers
                      */
  struct {
    int lowtype;
    int uptype;
    ISZ_T lowb;
    ISZ_T upb;
  } bounds[7];
  struct {       /* mark assumed size and adjustable arrays */
    int ndim;    /* number of dimensions */
    int assumsz; /*  0, not assumed size
                  *  1, assumed size
                  * >1, last dimension not assumed size
                  */
    int adjarr;  /*  0, not adjustable array
                  * >1, adjustable array
                  */
    int ndefer;  /* number of deferred dimensions (:) */
    ILM_T *ilmp; /* ilm pointer to ilms if adjustable array */
  } arrdim;
  int tkntyp;        /* token effecting semant reduction */
  struct {           /* atomic */
    int lineno;      /* line number of atomic */
    bool seen;       /* atomic directive just seen */
    bool pending;    /* atomic directive not yet applied */
    int action_type; /* (read|write|update|capture) */
  } atomic;
  int parallel;            /* parallel nesting level - PARALLEL, DOACROSS,
                            * PARALLELDO, PARALLELSECTIONS:
                            *  0 - not parallel
                            * >0 - parallel nesting level (1 => outermost)
                            */
  bool expect_do;          /* next statement after DOACROSS, PDO, or
                            * PARALLELDO needs to be a DO.
                            */
  bool close_pdo;          /* A DO loop for a PDO, PARALLELDO, or DOACROSS
                            * has been processed and its removal from the
                            * DOIF stack is delayed until the next
                            * statement is processed.  For PDO and
                            * PARALLELDO, the next statement may be the
                            * optional 'end' statement for the directive.
                            * For PDO, the decision to emit a barrier
                            * is also delayed since its ENDDO may specify
                            * NOWAIT.  For DOACROSS and PARALLELDO, the
                            * the parallel region is closed when the
                            * DO loop is closed.
                            */
  int sc;                  /* SC_LOCAL or SC_PRIVATE for temporaries */
  int ctemps;              /* avail counter for function value temps */
  int scope;               /* counter to keep track of the current scope
                            * for constructs which define a new scope
                            * (primarily, the parallel constructs):
                            *  0 - outermost (subprogram)
                            * >0 - scope nesting level
                            */
  SCOPESTACK *scope_stack; /* pushed/popped as scopes are entered/left */
  int scope_size;          /* size of scope stack */
  int threadprivate_dtype; /* dtype record used for the vector of pointers
                            * created for threadprivate common blocks.
                            */
  int it_dtype;            /* dtype record used for the mp run-time
                            * iteration data structure.
                            */
  int its_dtype;           /* dtype record used for the mp run-time
                            * iteration data structure.
                            */
  int blksymnum;
  bool ignore_default_none; /* don't perform the OMP DEFAULT(NONE) check */
  int collapse;             /* collapse value for the pardo or pdo */
  int collapse_depth;       /* depth of collapse loop; 1 => innermost */
  int task;                 /* depth of task
                             *  0 - not in task
                             * >0 - task nesting level (1 => outermost)
                             */
  /*
   * the following members are initialized to values which reflect the
   * default type for the extents and subscripts of arrays.  The type could
   * either be 32-int or 64-bit (BIGOBJects & -Mlarge_arrays).
   *
   */
  struct {
    int dtype; /* dtype used for the bound temps */
    int store; /* ILM opc for storing a bound value */
    int load;  /* ILM opc for loading a bound value */
    int mul;   /* ILM opc for multiplying */
    int sub;   /* ILM opc for substract */
    int add;   /* ILM opc for add */
    int con;   /* ILM opc for integer constants */
    int zero;  /* zero entry for zero */
    int one;   /* sym etnry for one */
  } bnd;
} SEM;

extern SEM sem;

/*
 * NTYPE - number of basic types; this must include the NCHARACTER
 * type even though it may not be an available feature.
 */
#define NTYPE 21

#define NOPC 14

extern short promote_ilms[NTYPE];
extern short ilm_opcode[NOPC][2][NTYPE + 1];
extern INT cast_types[NTYPE][2][2];

#define ILMA(n) (ilmb.ilm_base[n])

#define IS_COMPARE(opc) (opc >= IM_EQ && opc <= IM_GT)
#define IS_LOGICAL(opc)                                                        \
  (IS_COMPARE(opc) || (opc >= IM_LAND && opc <= IM_LOR) ||                     \
   (opc >= IM_AND64 && opc <= IM_AND) || (opc >= IM_OR64 && opc <= IM_OR) ||   \
   (opc >= IM_NOT64 && opc <= IM_LNOP) || opc == IM_LAND8 || opc == IM_LOR8 || \
   opc == IM_KAND || opc == IM_KOR || opc == IM_KNOT || opc == IM_LNOT8 ||     \
   opc == IM_LNOP8)
#define IS_INTRINSIC(st) (st == ST_INTRIN || st == ST_GENERIC || st == ST_PD)

#define INSIDE_STRUCT (sem.stsk_depth != 0)

#define GET_OPCODE(opc, dt) \
  (ilm_opcode[opc][(DTY(dt) == TY_ARRAY ? TRUE : FALSE)][DTYG(dt)])

#define DCLCHK(sptr)                                                          \
  if (sem.none_implicit && !DCLDG(sptr) && !E38G(sptr)) {                     \
    error(38, !XBIT(124, 0x20000) ? 3 : 2, gbl.lineno, SYMNAME(sptr), CNULL); \
    E38P(sptr, 1);                                                            \
  }

#define DOCHK(sptr) \
  if (DOVARG(sptr)) \
    error(115, 2, gbl.lineno, SYMNAME(sptr), CNULL);

/* if sp == 0, bound is '*' */
#define ILMBOUND(sp)                                   \
  (((sp) == 0)                                         \
       ? 0                                             \
       : (STYPEG(sp) == ST_CONST ? ad2ilm(IM_ICON, sp) \
                                 : ad2ilm(IM_ILD, ad2ilm(IM_BASE, sp))))

#define DPVAL(a) ad2ilm(IM_DPVAL, a)
#define DPREF(a) ad2ilm(IM_DPREF, a)
#define DPSCON(a) ad2ilm(IM_DPSCON, a)
#define DPNULL ad1ilm(IM_DPNULL)

void dmp_const(CONST *acl, int indent);

/*  declare external functions called only from within semant: */

void emit_epar(void); /* semsmp.c: */
void emit_bcs_ecs(int);
void end_parallel_clause(int);
void add_dflt_allocated(int);
void add_dflt_firstprivate(int, int);
INT chkcon();
INT const_fold(); /* semutil.c: */
ISZ_T chkcon_to_isz(struct sst *, bool);
INT chktyp();
INT chk_scalartyp();
INT chk_arr_extent();
int mkexpr();
int chkvar();
int add_base();
int chksubstr();
int get_temp(int);
int get_itemp(int);
int mkvarref();
int mklvalue(), mkmember();
int mklabelvar64(int);
bool is_varref();
void binop();
void mklogint4();
void link_members();
void chkstruct();
void assign();
void do_begin(DOINFO *, int, int, int);
void do_parbegin(DOINFO *, int, int, int);
void do_lastval(DOINFO *, int, int, int);
void do_end(DOINFO *);
void cngtyp();
void mklogint4();
void negate_const();
char *prtsst();
DOINFO *get_doinfo(int);

void chk_adjarr();
void gen_arrdsc(); /* semutil2.c: */
int mk_arrdsc();
void gen_allocate(int, int, int);
void gen_deallocate(int, int);
void sem_set_storage_class(int);
int enter_lexical_block(int);
void exit_lexical_block(int);
void dmp_doif(int);

int ad1ilm(int);
int ad2ilm(int, int);
int ad3ilm(int, int, int); /* ilmutil.c: */
int ad4ilm(int, int, int, int);
int ad5ilm(int, int, int, int, int);
void dumpilmtrees(void);
int lnegate();
void wrilms(int);
void add_ilms(ILM_T *);
void mkbranch(int, int, int);
void gwrilms(int nilms);
void fini_next_gilm(void);
void init_next_gilm(void);
void swap_next_gilm(void);
int rdilms(void);
void rewindilms(void);
#if DEBUG
/* FIXME those two functions do the same thing, also see _dumpilms */
void dmpilms(void);
void dumpilms(void);
#endif
ILM_T *save_ilms(int);
void dinit(VAR *ivl, CONST *ict); /* dinit.c */
bool dinit_ok(int);
void dmp_ivl(VAR *, FILE *);
void dmp_ict(CONST *, FILE *);
void semfin(); /* semfin.c */
int mklogopnd();
int ref_based_object(int);
int decl_private_sym(int);
void par_push_scope(bool);
void par_pop_scope(void);
int sem_check_scope(int, int);

/* semfunc.c */
int func_call();
int ref_intrin();
int ref_pd();
int mkarg();
int ref_stfunc();
int ref_entry();
int chkarg();
int select_gsame(int);
int mkipval(INT);
void subr_call();
void define_stfunc();

/* semutil0.c */
void semant_init(void);
void semant_reinit(void);

#endif // SEMANT_H_
