/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file
    \brief Semantic analyzer data definitions.
 */

/* Semantic stack entry types (SST_ID) */
#define S_NULL 0     /* empty/invalid */
#define S_CONST 1    /* scalar constant */
#define S_EXPR 2     /* expression */
#define S_LVALUE 3   /* non-whole variable */
#define S_LOGEXPR 4  /* (obsolete; was logical expression) */
#define S_STAR 5     /* * */
#define S_VAL 6      /* %VAL(expr) argument (VMS) */
#define S_IDENT 7    /* identifier, possibly a whole variable */
#define S_LABEL 8    /* label */
#define S_STFUNC 9   /* statement function reference */
#define S_REF 10     /* %REF(expr) argument (VMS)*/
#define S_TRIPLE 11  /* j:k:m */
#define S_KEYWORD 12 /* keyword, not an identifier */
#define S_ACONST 13  /* array constant */
#define S_SCONST 14  /* structure constant */
#define S_DERIVED 15 /* derived type object */

/* max pow is 10, num[] has 4 elements and 128 bits   */
#define POW1 1
#define POW2 2

#define RADIX2 2 /* Binary */
#define NOT_GET_VAL 0 /* 0 is default, This should be ruled out */
#define NO_REAL -5 /* if the processor supports no real type with radix RADIX  */
#define KEYWD_ARGS2 2

/* macros for checking to see if a derived type has a defined I/O
 * routine associated with it. DT_HAS_IO_FREAD true if derived type
 * has a formatted read associated with it. DT_HAS_IO_UREAD true
 * if derived type has an unformatted read associated with it.
 * DT_HAS_IO_FWRITE true if derived type has a formatted write
 * associated with it. DT_HAS_IO_UWRITE true if derived type has an
 * unformatted write associated with it. DT_HAS_IO true if there's any
 * defined I/O subroutine associated with it.
 */
#define DT_IO_UNKNOWN 0x0 /* Not yet known if derived type has defined I/O */
#define DT_IO_NONE 0x1    /* No defined I/O for derived type */
#define DT_IO_FREAD 0x2   /* Defined READ(FORMATTED) */
#define DT_IO_UREAD 0x4   /* Defined READ(UNFORMATTED) */
#define DT_IO_FWRITE 0x8  /* Defined WRITE(FORMATTED) */
#define DT_IO_UWRITE 0x10 /* Defined WRITE(UNFORMATTED) */

#define DT_HAS_IO_FREAD(dt) (((UFIOG(DTY(dt + 3)) & DT_IO_FREAD)))
#define DT_HAS_IO_UREAD(dt) (((UFIOG(DTY(dt + 3)) & DT_IO_UREAD)))
#define DT_HAS_IO_FWRITE(dt) (((UFIOG(DTY(dt + 3)) & DT_IO_FWRITE)))
#define DT_HAS_IO_UWRITE(dt) (((UFIOG(DTY(dt + 3)) & DT_IO_UWRITE)))
#define DT_HAS_IO(dt) ((UFIOG(DTY(dt + 3))))

#define NEW_INTRIN \
  65535 /* (newer) intrinsic with no predefined entry in the symtab */

/*
 * Generate lexical block debug information?  Criteria:
 *    -debug
 *    not -Mvect   (!flg.vect)
 *    not -Mconcur (!XBIT(34,0x200)
 *    lex block disabled (!XBIT(123,0x4000))
 */
#define DBG_LEXBLK \
  flg.debug && !flg.vect && !XBIT(34, 0x200) && !XBIT(123, 0x400)

typedef struct type_list {
  LOGICAL is_class;
  int dtype;
  int label;
  struct type_list *next;
} TYPE_LIST;

typedef struct {
  int first;
  int last;
} FLITM;

typedef struct _std_range {
  int start;
  int mid;
  int end;
  struct _std_range *next;
} STD_RANGE;

typedef struct _std_record {
  struct sst *stkp;      /* the value of the semantic stack for the <elp> */
  int std;               /* the STD_LAST when meeting the <elp> */
  struct _std_record *next;
} STD_RECORD;

typedef struct xyyz {
  struct xyyz *next;
  int ast;
  union {
    int sptr;
    int ilm;
    int cltype;
    INT conval;
    struct sst *stkp;
    FLITM *flitmp;
  } t;
} ITEM;
#define ITEM_END ((ITEM *)1)

typedef enum {
  LP_PDO = 1,         /* omp do */
  LP_PARDO,           /* parallel do */
  LP_DISTRIBUTE,      /* distribute loop: distribute construct */
  LP_DIST_TEAMS,      /* distribute loop: teams distribute construct */
  LP_DIST_TARGTEAMS,  /* distribute loop: target teams distribute construct */
  LP_DISTPARDO,       /* distribute loop: distribute parallel do ... */
  LP_DISTPARDO_TEAMS, /* distribute loop: teams distribute parallel do ... */
  LP_DISTPARDO_TARGTEAMS, /* distribute loop: target teams dist... */
  LP_PARDO_OTHER,         /* parallel do: created for any distribute parallel do
                           * construct. */
} distlooptype;

typedef struct {
  int index_var;    /* do index variable */
  int init_expr;
  int limit_expr;
  int step_expr;
  int count;        /* loop count ast */
  int lastval_var;
  int collapse;     /* collapse level if applicable; 1 is innermost */
  char prev_dovar;  /* DOVAR flag of index variable before it's entered */
  LOGICAL nodepchk;
  distlooptype distloop;
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
  LOGICAL is_dovar;
} NOSCOPE_SYM;

typedef enum { // CAUTION: order in di_name in semutil2.c must match
  DI_IF,
  DI_IFELSE,
  DI_DO,
  DI_DOCONCURRENT,
  DI_DOWHILE,
  DI_WHERE,
  DI_ELSEWHERE,
  DI_FORALL,
  DI_CASE,
  DI_SELECT_TYPE,
  DI_ASSOC,
  DI_BLOCK,

  // Place non-directives above; directives below this point.
  DI_FIRST_DIRECTIVE,
  DI_PAR = DI_FIRST_DIRECTIVE,
  DI_PARDO,
  DI_PDO,
  DI_DOACROSS,
  DI_PARSECTS,
  DI_SECTS,
  DI_SINGLE,
  DI_CRITICAL,
  DI_MASTER,
  DI_ORDERED,
  DI_WORKSHARE,
  DI_PARWORKS,
  DI_TASK,
  DI_ATOMIC_CAPTURE,
  DI_SIMD,
  DI_TASKGROUP,
  DI_TASKLOOP,
  DI_TARGET,
  DI_TARGETENTERDATA,
  DI_TARGETEXITDATA,
  DI_TARGETDATA,
  DI_TARGETUPDATE,
  DI_DISTRIBUTE,
  DI_TEAMS,
  DI_DECLTARGET,
  DI_DISTPARDO,
  DI_TARGPARDO,
  DI_TARGETSIMD,
  DI_TARGTEAMSDIST,
  DI_TEAMSDIST,
  DI_TARGTEAMSDISTPARDO,
  DI_TEAMSDISTPARDO,
  DI_CUFKERNEL,
  DI_ACCREG,
  DI_ACCKERNELS,
  DI_ACCPARALLEL,
  DI_ACCDO,
  DI_ACCLOOP,
  DI_ACCREGDO,
  DI_ACCREGLOOP,
  DI_ACCKERNELSDO,
  DI_ACCKERNELSLOOP,
  DI_ACCPARALLELDO,
  DI_ACCPARALLELLOOP,
  DI_ACCKERNEL,
  DI_ACCDATAREG,
  DI_ACCHOSTDATA,
  DI_ACCSERIAL,
  DI_ACCSERIALLOOP,
  DI_MAXID, // CAUTION: DI_MAXID cannot be greater than 63 (see DI_NEST)
} DI_KIND;

typedef struct {               // construct (DO, IF, etc.) stack entries
  DI_KIND Id;
  int lineno;                  // beginning line number of control structure
  BIGUINT nest;                // bit vector of structures present in the stack,
                               // including the current structure
  int name;                    // construct name (may be 0)
  int exit_label;              // for DO, target label of an EXIT stmt
                               // for IF and SELECT, label of statement after
                               // END IF or END SELECT (may be 0)

  // OpenMP fields used in non-OpenMP slots
  NOSCOPE_SYM *no_scope_base;  // list of unscoped variables
  int no_scope_avail;
  int no_scope_size;
  int no_scope_forall;

  union {
    struct {                   // DO
      int do_label;            // DO statement label
      int cycle_label;         // target label for CYCLE statement
      int top_label;           // label of the top of a DO while loop
      int ast;                 // DO ast
      DOINFO *doinfo;          // DO info record
                               // remaining fields are for DO CONCURRENT loops;
                               // some fields are only set on an innermost loop
      SPTR symavl;             // stb.stg_avail value at entry (sym "watermark")
      int count;               // var=triplet control count -- outermost=1
      int kind;                // temp: 1) curr locality kind; 2) component kind
      bool no_default;         // loop has a DEFAULT(NONE) locality spec?
      int popindex;            // index symbol to pop
      int block_sym;           // loop body block sym
      int syms;                // list of index, local, local_init, shared syms
      int last_sym;            // last sym in syms list
      int label_syms;          // list of label syms
      int error_syms;          // list of syms that have errors
      int mask_std;            // mask std (may be null)
      int body_std;            // first loop body std (may be null)
    } u1;
    struct {                   // WHERE
      int shapedim;            // number of dimensions in the WHERE construct
      int masked;              // masked elsewhere
    } u2;
    struct {                   // SELECT CASE
      int case_expr;           // SELECT CASE expression AST
      int dtype;               // data type of (valid) SELECT CASE expression
      int beg_default;         // last STD generated when 'CASE DEFAULT' is
                               // parsed (STD immediately preceding first STD
                               // generated for the default block); if the
                               // CASE DEFAULT appears before a CASE, the STDs
                               // of the default block are 'saved' and this
                               // field locates the first STD of the default
                               // block; initially 0
      int end_default;         // if the default block is saved, this is the
                               // last STD of the default block; initially 0
                               // to determine:
                               //  - if the default block is empty:
                               //        beg_default == 0
                               //  - if CASE DEFAULT immediately precedes a
                               //    CASE or END SELECT:
                               //        beg_default != 0 && end_default == 0
                               //  - if CASE DEFAULT has been saved (and
                               //    appears before a CASE):
                               //        beg_default != 0 && end_default != 0
      int swel_hd;             // index to the start of the SWEL list for CASEs
      int allo_chtmp;          // allocated char temp for case expr if necessary
      char default_seen;       // non-zero if CASE DEFAULT is present
      char default_complete;   // non-zero if CASE DEFAULT processing is done
      char pending;            // non-zero if CASE block processing is not done
    } u3;
    struct {                   // OpenMP
      int bpar;                // A_MP_PARALLEL ast for PARALLEL and combo
                               // directives; sptr of global semaphore variable
                               // created for CRITICAL <ident>
      int beginp;              // parallel construct beginning ast; bpar and
                               // beginp fields are both set for combo dirs
      int target;              // OpenMP use
      int teams;               // OpenMP use
      int distribute;          // OpenMP use
      REDUC *reduc;            // reductions for parallel constructs
      REDUC_SYM *lastprivate;  // lastprivate for parallel constructs
      ITEM *allocated;         // list of allocated private variables
      ITEM *region_vars;       // accelerator region copy/local/mirror vars
      union {
        struct {               // OpenMP - DO
          int sched_type;      // one of DI_SCHxxx
          int chunk;           // sptr for chunk size (0 if absent)
          LOGICAL is_ordered;  // loop has the ordered attribute?
          LOGICAL is_simd;     // loop can be simd loop?
          int dist_chunk;      // sptr for distribute chunk size (0 if absent)
        } v1;
        struct {               // OpenMP - SECTIONS
          int sect_cnt;        // number of SECTION blocks
          int sect_var;        // where to store section number
        } v2;
      } v;
    } u4;
    struct {                   // SELECT TYPE
      int selector;            // sptr of selector
      LOGICAL is_whole;        // whether selector is a whole variable
      int active_sptr;         // sptr of active temp pointer
      int beg_std;             // std of SELECT TYPE stmt
      int end_select_label;    // sptr of label for end SELECT stmt
      int class_default_label; // sptr of label to class default
      TYPE_LIST *types;        // list of types
    } u5;
    struct {                   // FORALL
      int laststd;             // last std at start of forall processing
      SPTR symavl;             // stb.stg_avail value at entry (sym "watermark")
      DTYPE dtype;             // explicit index data type
      int idxlist;             // list of index var sptrs
      int forall_ast;
    } u6;
    struct {                   // ASSOCIATE
      ITEM *sptrs;             // ASSOCIATE names
    } u7;
    struct {                   // BLOCK
      int encl_block_scope;    // scope index of enclosing block (if any)
    } u8;
  } u;
} DOIF;

#define DI_SCH_STATIC 0
#define DI_SCH_DYNAMIC 1
#define DI_SCH_GUIDED 2
#define DI_SCH_INTERLEAVE 3
#define DI_SCH_RUNTIME 4
#define DI_SCH_AUTO 5
#define DI_SCH_DIST_STATIC 6
#define DI_SCH_DIST_DYNAMIC 7

#define DI_ID(d) sem.doif_base[d].Id
#define DI_LINENO(d) sem.doif_base[d].lineno
#define DI_NEST(d) sem.doif_base[d].nest
#define DI_NAME(d) sem.doif_base[d].name
#define DI_EXIT_LABEL(d) sem.doif_base[d].exit_label

#define DI_DO_LABEL(d) sem.doif_base[d].u.u1.do_label
#define DI_CYCLE_LABEL(d) sem.doif_base[d].u.u1.cycle_label
#define DI_TOP_LABEL(d) sem.doif_base[d].u.u1.top_label
#define DI_DO_AST(d) sem.doif_base[d].u.u1.ast
#define DI_DOINFO(d) sem.doif_base[d].u.u1.doinfo
#define DI_DO_POPINDEX(d) sem.doif_base[d].u.u1.popindex
#define DI_CONC_SYMAVL(d) sem.doif_base[d].u.u1.symavl
#define DI_CONC_COUNT(d) sem.doif_base[d].u.u1.count
#define DI_CONC_KIND(d) sem.doif_base[d].u.u1.kind
#define DI_CONC_NO_DEFAULT(d) sem.doif_base[d].u.u1.no_default
#define DI_CONC_BLOCK_SYM(d) sem.doif_base[d].u.u1.block_sym
#define DI_CONC_SYMS(d) sem.doif_base[d].u.u1.syms
#define DI_CONC_LAST_SYM(d) sem.doif_base[d].u.u1.last_sym
#define DI_CONC_LABEL_SYMS(d) sem.doif_base[d].u.u1.label_syms
#define DI_CONC_ERROR_SYMS(d) sem.doif_base[d].u.u1.error_syms
#define DI_CONC_MASK_STD(d) sem.doif_base[d].u.u1.mask_std
#define DI_CONC_BODY_STD(d) sem.doif_base[d].u.u1.body_std

#define DI_SHAPEDIM(d) sem.doif_base[d].u.u2.shapedim
#define DI_MASKED(d) sem.doif_base[d].u.u2.masked

#define DI_CASE_EXPR(d) sem.doif_base[d].u.u3.case_expr
#define DI_DTYPE(d) sem.doif_base[d].u.u3.dtype
#define DI_BEG_DEFAULT(d) sem.doif_base[d].u.u3.beg_default
#define DI_END_DEFAULT(d) sem.doif_base[d].u.u3.end_default
#define DI_SWEL_HD(d) sem.doif_base[d].u.u3.swel_hd
#define DI_ALLO_CHTMP(d) sem.doif_base[d].u.u3.allo_chtmp
#define DI_DEFAULT_SEEN(d) sem.doif_base[d].u.u3.default_seen
#define DI_DEFAULT_COMPLETE(d) sem.doif_base[d].u.u3.default_complete
#define DI_PENDING(d) sem.doif_base[d].u.u3.pending

#define DI_BPAR(d) sem.doif_base[d].u.u4.bpar
#define DI_BTARGET(d) sem.doif_base[d].u.u4.target
#define DI_BTEAMS(d) sem.doif_base[d].u.u4.teams
#define DI_BDISTRIBUTE(d) sem.doif_base[d].u.u4.distribute
#define DI_CRITSYM(d) sem.doif_base[d].u.u4.bpar
#define DI_BEGINP(d) sem.doif_base[d].u.u4.beginp
#define DI_REDUC(d) sem.doif_base[d].u.u4.reduc
#define DI_LASTPRIVATE(d) sem.doif_base[d].u.u4.lastprivate
#define DI_ALLOCATED(d) sem.doif_base[d].u.u4.allocated
#define DI_REGIONVARS(d) sem.doif_base[d].u.u4.region_vars
#define DI_SCHED_TYPE(d) sem.doif_base[d].u.u4.v.v1.sched_type
#define DI_CHUNK(d) sem.doif_base[d].u.u4.v.v1.chunk
#define DI_DISTCHUNK(d) sem.doif_base[d].u.u4.v.v1.dist_chunk
#define DI_IS_ORDERED(d) sem.doif_base[d].u.u4.v.v1.is_ordered
#define DI_ISSIMD(d) sem.doif_base[d].u.u4.v.v1.is_simd
#define DI_SECT_CNT(d) sem.doif_base[d].u.u4.v.v2.sect_cnt
#define DI_SECT_VAR(d) sem.doif_base[d].u.u4.v.v2.sect_var
#define DI_NOSCOPE_BASE(d) sem.doif_base[d].no_scope_base
#define DI_NOSCOPE_SIZE(d) sem.doif_base[d].no_scope_size
#define DI_NOSCOPE_AVL(d) sem.doif_base[d].no_scope_avail
#define DI_NOSCOPE_FORALL(d) sem.doif_base[d].no_scope_forall

#define DI_SELECTOR(d) sem.doif_base[d].u.u5.selector
#define DI_IS_WHOLE(d) sem.doif_base[d].u.u5.is_whole
#define DI_ACTIVE_SPTR(d) sem.doif_base[d].u.u5.active_sptr
#define DI_END_SELECT_LABEL(d) sem.doif_base[d].u.u5.end_select_label
#define DI_TYPE_BEG(d) sem.doif_base[d].u.u5.beg_std
#define DI_CLASS_DEFAULT_LABEL(d) sem.doif_base[d].u.u5.class_default_label
#define DI_SELECT_TYPE_LIST(d) sem.doif_base[d].u.u5.types

#define DI_FORALL_LASTSTD(d) sem.doif_base[d].u.u6.laststd
#define DI_FORALL_SYMAVL(d) sem.doif_base[d].u.u6.symavl
#define DI_FORALL_DTYPE(d) sem.doif_base[d].u.u6.dtype
#define DI_IDXLIST(d) sem.doif_base[d].u.u6.idxlist
#define DI_FORALL_AST(d) sem.doif_base[d].u.u6.forall_ast

#define DI_ASSOCIATIONS(d) sem.doif_base[d].u.u7.sptrs

#define DI_ENCL_BLOCK_SCOPE(d) sem.doif_base[d].u.u8.encl_block_scope

#define onel 1ULL
#define DI_B(t) (onel << (t))
#define DI_IN_NEST(d, t) (d && d <= sem.doif_depth && (DI_NEST(d) & DI_B(t)))

#define NEED_DOIF(df, typ)                                               \
  {                                                                      \
    df = ++sem.doif_depth;                                               \
    NEED(df + 1, sem.doif_base, DOIF, sem.doif_size, sem.doif_size + 8); \
    BZERO(sem.doif_base+df, DOIF, 1);                                    \
    DI_LINENO(df) = gbl.lineno;                                          \
    DI_ID(df) = typ;                                                     \
    DI_NEST(df) = DI_NEST(df - 1) | DI_B(typ);                           \
  }

/* Define Initializer Variable Tree */
typedef struct var_init { /* used for elements of dinit variable list */
  short id;
#define Dostart 0
#define Doend 1
#define Varref 2
  union {
    struct {
      int indvar; /* ast */
      int lowbd;  /* ast */
      int upbd;   /* ast */
      int step;   /* ast */
    } dostart;
    struct {
      struct var_init *dostart;
    } doend;
    struct {
      /* Semantic stack info for variable reference */
      int id;
      int ptr; /* ast */
      int dtype;
      int shape;
      struct var_init *subt; /* for derived-type: points to
                     var list of mangled-name structure members */
    } varref;
  } u;
  struct var_init *next;
} VAR;

/* typedef and macros to access array constructor lists: */
/* also used for structure constructors */
/* NOW also used for initialization list of constants which
    formerly used the CONST structure */

/* NOTE: repeatc may not be needed if it can be recalculated from sptr
         field.  sptr field may be able to be relocated to u2.
 */
typedef struct _aexpr AEXPR;
typedef struct _acl {
  char id;               /* one of AC_... */
  unsigned is_const : 1, /* is it constant ? */
      ci_exprt : 1,      /* 1==>component initialization has been exported */
      no_dinitp : 1;     /* do not set DINIT flag */
  DTYPE dtype;           /* used in init. Later if AC_ACONST or AC_SCONST */
  DTYPE ptrdtype;        /* ptr type if pointer init */
  int repeatc;           /* used in init. ast or ==0 for default of 1 */
  int sptr;              /* used for DATA stmt, VMS struct inits, and F95
                          * derived type component initializers */
  int size;              /* set by chk_constructor() - the ast of the size
                          * (upper bound) of the temporary (if AC_ACONST).
                          */
  INT conval;            /* "constant" value when evaluating F95
                          * derived type component initializations
                          * for non-static variable */
  struct _acl *next;     /* next in list */
  struct _acl *subc;     /* down in tree. Valid for AC_ACONST,
                               AC_SCONST, AC_IDO, AC_REPEAT ,
                               AC_VMSSTRUCT, AC_VMSUNION */
  union {
    struct sst *stkp; /* if AC_EXPR   */
    AEXPR *expr;      /* if AC_AEXPR */
    int ast;          /* if AC_AST, AC_CONST, AC_IDENT */
    int i;            /* if AC_ICONST */
    INT count;        /* if AC_REPEAT */
    DOINFO *doinfo;   /* if AC_IDO    */
  } u1;
  union {
    int array_i; /* if AC_EXPR, AC_AST */
    STD_RANGE *std_range; /* if AC_IDO */
  } u2;
} ACL;

struct _aexpr {
  int op;
  ACL *lop;
  ACL *rop;
};

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
#define AC_ICONST 14
/* integer constant value, currently used to keep
 * intrinsic routine selector
 */
#define AC_CONVAL 15
/* Type of ACL leaf item generated by calling
 * eval_init_expr/eval_init_expr_item. The conval
 * field contains the results of the evaluation.
 * The type of the value is a literal constant if
 * the type a TY_WORD. Otherwise, the value is the
 * sptr of a constant.
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

typedef enum {
  AC_I_NONE = 0,
  AC_I_adjustl,
  AC_I_adjustr,
  AC_I_char,
  AC_I_ichar,
  AC_I_index,
  AC_I_int,
  AC_I_ishft,
  AC_I_ishftc,
  AC_I_kind,
  AC_I_lbound,
  AC_I_len,
  AC_I_len_trim,
  AC_I_nint,
  AC_I_null,
  AC_I_repeat,
  AC_I_reshape,
  AC_I_scan,
  AC_I_selected_int_kind,
  AC_I_selected_real_kind,
  AC_I_size,
  AC_I_transfer,
  AC_I_trim,
  AC_I_ubound,
  AC_I_verify,
  AC_I_shape,
  AC_I_min,
  AC_I_max,
  AC_I_fltconvert,
  AC_I_floor,
  AC_I_ceiling,
  AC_I_mod,
  AC_I_sqrt,
  AC_I_exp,
  AC_I_log,
  AC_I_log10,
  AC_I_sin,
  AC_I_cos,
  AC_I_tan,
  AC_I_asin,
  AC_I_acos,
  AC_I_atan,
  AC_I_atan2,
  AC_I_selected_char_kind,
  AC_I_abs,
  AC_I_iand,
  AC_I_ior,
  AC_I_ieor,
  AC_I_merge,
  AC_I_lshift,
  AC_I_rshift,
  AC_I_maxloc,
  AC_I_maxval,
  AC_I_minloc,
  AC_I_minval,
  AC_I_scale,
  AC_I_transpose,
} AC_INTRINSIC;

#define BINOP(p) ((p)->op != AC_NEG && (p)->op != AC_CONV)

/* getitem() AREA's */
#define ACL_AREA 0
#define ACL_SAVE_AREA 3

#define GET_ACL(a) get_acl(a)

typedef struct {   /* STRUCTURE stack entries */
  char type;       /* 's': STRUCTURE; 'u': UNION; 'm': MAP; 'd': derived type */
  char mem_access; /* 0 - public by default, 'v'=>access private */
  int sptr;        /* Sym ptr to field name list having this structure */
  int dtype;       /* Pointer to structure dtype */
  int last;        /* last member; updated by link_members */
  ACL *ict_beg;    /* Initializer Constant Tree begin */
  ACL *ict_end;    /* Initializer Constant Tree end */
} STSK;
/* access entries in STRUCTURE stack; 0 ==> top of stack, 1 ==> 1 back, etc. */
#define STSK_ENT(i) sem.stsk_base[sem.stsk_depth - (i)-1]

typedef struct equiv_var { /* variable references in EQUIVALENCE statements */
  int sptr;
  int lineno;
  int ps;
  int substring; /* ast of left substring index, 0=>none */
  int subscripts;
  ISZ_T byte_offset;
  /* the next field can be made smaller if more fields must be added */
  INT is_first; /* nonzero marks first in a group */
  int next;
} EQVV;

#define EQV(i) sem.eqv_base[i]
#define EQV_NUMSS(i) sem.eqv_ss_base[i]         /* number of subscripts */
#define EQV_SS(i, j) sem.eqv_ss_base[i + j + 1] /* j from 0 to EQV_NUMSS(i) */

typedef struct _seql { /* variable references in [NO]SEQUENCE statements */
  char type;           /* 's' - SEQUENCE; 'n' - NOSEQUENCE */
  int sptr;            /*  sym ptr of object in statement */
  struct _seql *next;  /*  next _seql item */
} SEQL;

typedef struct _accl { /* variable references in ACCESS statements */
  char type;           /* 'u' - PUBLIC; 'v' - PRIVATE */
  char oper;           /* 'o' - operator; ' ' - not an operator */
  int sptr;            /*  sym ptr of object in statement */
  struct _accl *next;  /*  next _accl item */
} ACCL;

/*
 * For saving state when interface blocks are processed.
 */
typedef struct {
  int currsub; /* previous subprogram */
  RU_TYPE rutype;  /* type of previous subprogram */
  bool module_procedure; /* instantiated with MODULE PROCEDURE <id> */
  int pgphase;
  int none_implicit; /* bit vector indicating presence of implicit
                      * none.  A nonzero value indicates that all
                      * variables need to be declared.  Bit values
                      * indicate placement of the implicit none:
                      *
                      * 0x00 - not present
                      * 0x01 - -dclchk specified on the command line
                      * 0x02 - present in the host
                      * 0x04 - present in a contained procedure
                      *
                      * NOTE: the latter two values correspond to the
                      * possible values of semant.c:host_present.
                      */
  LOGICAL seen_implicit;
  LOGICAL seen_parameter;
  int generic;     /* if non-zero, sptr of ST_GENERIC */
  int operator;    /* if non-zero, sptr of ST_OPERATOR */
  char gnr_rutype; /* routine type of the generic */
  char optype;     /* 0x00 - defined operator
                    * 0x01 - unary intrinsic operator
                    * 0x02 - binary intrinsic operator
                    * 0x03 - unary and binary intrinsic operator
                    * 0x04 - assignment
                    */
  char abstract;   /* if nonzero, ABSTRACT INTERFACE */
  char opval;      /* OP_... value if intrinsic operator */
  int hpfdcl;      /* available index for the hpf declarations whose
                    * semantic processing is deferred.  For the interface,
                    * the hpf declarations are saved as indices
                    * [hpfdcl, sem.hpfdcl-1].
                    */
  int nml;         /* copy of sem.nml */
} INTERF;

/*
 * support for extracting positional or keyword arguments for an
 * intrinsic:
 * 1.  for each intrinsic, the symbol table utility has created a string
 *     which defines the arguments and their positions in the argument
 *     list and keywords.
 *
 *     Optional arguments are indicating by prefixing their keywords with
 *     '*'.
 * 2.  get_kwd_args() extracts the arguments from the semantic list
 *     and places them in the static array argpos[] in positional
 *     order.  evl_kwd_args calls get_kwd_args() and evaluates each
 *     argument.
 */

typedef struct {
  struct sst *stkp; /* semantic stack entry for argument, possibly NULL */
  int ast;          /* its ast, possibly 0 */
} argpos_t;

#define ARG_STK(i) sem.argpos[i].stkp
#define ARG_AST(i) sem.argpos[i].ast
#define XFR_ARGAST(i) ARG_AST(i) = SST_ASTG(ARG_STK(i))

/*
 * define stack entry for evaluating implied do's in dinits and array
 * constructors.
 */
typedef struct {
  int sptr;
  INT currval;
  INT upbd;
  INT step;
} DOSTACK;
#define MAX_DOSTACK 8

/* List element type of a symbol in a SHARED parallel scoping clause. */
typedef struct scope_sym_tag {
  int sptr;  /* symbol appearing in the SHARED clause */
  int scope; /* its outer scope value */
  struct scope_sym_tag *next;
} SCOPE_SYM;

// "NORMAL" scopes are somewhat odd.  They are usually empty, and usually
// paired with a module, subprogram, block, or other scope, which contains
// user symbols.  One use of a NORMAL scope is to serve as a "marker" scope for
// popping multiple intervening (typically USE) scopes from the scope stack.
// As an exception (for whatever reason), a NORMAL scope is used to contain
// compiler created symbols at the end of a subprogram or module scope.

typedef enum SCOPEKIND {
  SCOPE_OUTER,            // global symbols (level 0)
  SCOPE_NORMAL,           // "marker/administrative/miscellaneous" scope
  SCOPE_MODULE,           // module or submodule
  SCOPE_SUBPROGRAM,       // external, internal, module, or interface subprogram
  SCOPE_BLOCK,
  SCOPE_INTERFACE,
  SCOPE_USE,
  SCOPE_PAR,
} SCOPEKIND;

// scope stack - indexed by sem.scope_level
//
// The optional sptr "id" for a scope is typically the enclosing program
// unit symbol.  For interface scopes, it is a value less than zero.
// A scope is "closed" if symbols from ancestor scopes are not visible,
// and "open" if they are visible.

typedef struct {
  SCOPEKIND kind;
  SPTR sptr;              // optional scope identifier; usually a symbol
  bool closed;            // symbols in containing scopes are inaccessible?
  bool Private;           // symbols are private?
  int symavl;             // "watermark" stb.stg_avail at scope entry
  int except;             // SYMI list of names not used
  int only;               // SYMI list of public names in a private scope
  int import;             // SYMI list of names explictly imported from host

  // parallel scope fields
  int rgn_scope;          // containing parallel region's scope entry index
  int par_scope;          // default parallel scope (PAR_SCOPE_SHARED ...)
  int end_prologue;       // end of parallel or task construct prologue;
                          // default(firstprivate) assignment insertion point
  int di_par;             // DOIF index for this scope
  int blk_sym;            // ST_BLOCK defining this scope
  int autobj;             // list of automatic data objects for this scope
  SC_KIND prev_sc;        // previous storage class
  SCOPE_SYM *shared_list; // PAR_SCOPE_SHARED shared symbols
  int mpscope_sptr;       // scope ST_BLOCK of the next scope (next scope sym)
  int uplevel_sptr;       // uplevel ST_BLOCK of the next scope, which keeps
                          // info for shared variables in the current region
} SCOPESTACK;

/* default scope of new symbols within a parallel region */
#define PAR_SCOPE_NONE 0
#define PAR_SCOPE_SHARED 1
#define PAR_SCOPE_PRIVATE 2
#define PAR_SCOPE_FIRSTPRIVATE 3
#define PAR_SCOPE_TASKNODEFAULT 4

#define BLK_SYM(i) sem.scope_stack[i].blk_sym
#define BLK_AUTOBJ(i) sem.scope_stack[i].autobj
#define BLK_UPLEVEL_SPTR(i) sem.scope_stack[i].uplevel_sptr
#define BLK_SCOPE_SPTR(i) sem.scope_stack[i].mpscope_sptr

/* scopestack.c */
void scopestack_init(void);
SCOPESTACK *curr_scope(void);
SCOPESTACK *get_scope(int index);
int get_scope_level(SCOPESTACK *scope);
SCOPESTACK *next_scope(SCOPESTACK *scope);
SCOPESTACK *next_scope_sptr(SCOPESTACK *scope, int sptr);
SCOPESTACK *next_scope_kind(SCOPESTACK *scope, SCOPEKIND kind);
SCOPESTACK *next_scope_kind_sptr(SCOPESTACK *scope, SCOPEKIND kind, int sptr);
SCOPESTACK *next_scope_kind_symname(SCOPESTACK *scope, SCOPEKIND kind,
                                    const char *symname);
int have_use_scope(int sptr);
LOGICAL is_except_in_scope(SCOPESTACK *scope, int sptr);
LOGICAL is_private_in_scope(SCOPESTACK *scope, int sptr);
void push_scope_level(int sptr, SCOPEKIND kind);
void pop_scope_level(SCOPEKIND kind);
void save_scope_level(void);
void restore_scope_level(void);
void par_push_scope(LOGICAL bind_to_outer);
void par_pop_scope(void);
#if DEBUG
void dumpscope(FILE *f);
void dump_one_scope(int sl, FILE *f);
#endif

/* module.c */
void allocate_refsymbol(int);
void set_modusename(int, int);
void use_init(void);
void init_use_stmts(void);
void add_use_stmt(void);
void add_submodule_use(void);
SPTR add_use_rename(SPTR, SPTR, LOGICAL);
void apply_use_stmts(void);
void add_isoc_intrinsics(void);
void open_module(SPTR);
void close_module(void);
void mod_combined_name(const char *);
void mod_combined_index(const char *);
SPTR begin_module(SPTR);
SPTR begin_submodule(SPTR, SPTR, SPTR, SPTR *);
LOGICAL get_seen_contains(void);
void mod_implicit(int, int, int);
void begin_contains(void);
void end_module(void);
LOGICAL has_cuda_data(void);
void mod_fini(void);
void mod_init(void);
int mod_add_subprogram(int);
void mod_end_subprogram(void);
void mod_end_subprogram_two(void);

/* semantio.c */
int get_nml_array(int);

typedef struct {
  struct _sem_bounds {
    int lowtype;
    int uptype;
    ISZ_T lowb;
    ISZ_T upb;
    int lwast;
    int upast;
  } bounds[MAXDIMS];
  struct _sem_arrdim { /* communicate info for <dim spec>s to mk_arrdsc() */
    int ndim;          /* number of dimensions */
    int ndefer;        /* number of deferred dimensions (:) */
    bool assumedrank;  /* TRUE if it's assumed rank (..) */
  } arrdim;
} SEM_DIM_SPECS;

/* semutil2.c */
ISZ_T size_of_array(DTYPE);
DTYPE chk_constructor(ACL *, DTYPE);
void chk_struct_constructor(ACL *);
void gen_type_initialize_for_sym(SPTR, int, int, DTYPE);
void clean_struct_default_init(int);
void restore_host_state(int);
void restore_internal_subprograms(void);
void dummy_program(void);
int have_module_state(void);
void fix_type_param_members(SPTR, DTYPE);
void add_type_param_initialize(int);
void add_p_dealloc_item(int sptr);
int gen_finalization_for_sym(int sptr, int std, int memAst);
void gen_alloc_mem_initialize_for_sym(int sptr, int std);
int add_parent_to_bounds(int parent, int ast);
int fix_mem_bounds2(int, int);
int fix_mem_bounds(int parent, int mem);
int init_sdsc(int, DTYPE, int, int);
int init_sdsc_bounds(SPTR, DTYPE, int, SPTR, int, int);
void save_module_state1(void);
void save_module_state2(void);
void restore_module_state(void);
void reset_module_state(void);
LOGICAL is_alloc_ast(int);
LOGICAL is_alloc_std(int);
LOGICAL is_dealloc_ast(int);
LOGICAL is_dealloc_std(int);
SPTR get_dtype_init_template(DTYPE);
void module_must_hide_this_symbol(int sptr);
void reset_internal_subprograms(void);
int mp_create_bscope(int reuse);
void save_struct_init(ACL *);
void chk_adjarr(void);
int mp_create_escope(void);
void dup_struct_init(int, int);
void gen_derived_type_alloc_init(ITEM *);
void save_host_state(int); /* semtutil2.c */
void add_auto_finalize(int);
int runtime_array(int);
DTYPE mk_arrdsc(void);
int gen_defer_shape(int, int, int);
ACL *eval_init_expr(ACL *e);
void gen_allocate_array(int);
void gen_deallocate_arrays(void);
void mk_defer_shape(SPTR);
void mk_assumed_shape(SPTR);
SPTR get_arr_const(DTYPE);
DTYPE select_kind(DTYPE, int, INT);
SPTR get_param_alias_var(SPTR, DTYPE);
void init_named_array_constant(int, int);
int init_sptr_w_acl(int, ACL *);
int init_derived_w_acl(int, ACL *);
ACL *mk_init_intrinsic(AC_INTRINSIC);
ACL *get_acl(int);
ACL *save_acl(ACL *);
ACL *construct_acl_from_ast(int, DTYPE, int);
ACL *rewrite_acl(ACL *, DTYPE, int);
ACL *all_default_init(DTYPE);
void dmp_acl(ACL *, int);
void mk_struct_constr(int, int);
void process_struct_components(int, void (*)(int));
int get_struct_leafcnt(int);
int get_first_mangled(int);
void re_struct_constr(int, int);
void propagate_attr(int, int);
ACL *dinit_struct_vals(ACL *, DTYPE, SPTR);
SPTR get_temp(DTYPE);
DTYPE get_temp_dtype(DTYPE, int);
SPTR get_itemp(DTYPE);
SPTR get_arr_temp(DTYPE, LOGICAL, LOGICAL, LOGICAL);
SPTR get_adjlr_arr_temp(DTYPE);
int get_shape_arr_temp(int);
SPTR get_ch_temp(DTYPE);
int need_alloc_ch_temp(DTYPE);
int sem_strcmp(const char *, const char *);
LOGICAL sem_eq_str(int, const char *);
void add_case_range(int, int, int);
int _i4_cmp(int, int);
int _i8_cmp(int, int);
int _char_cmp(int, int);
int _nchar_cmp(int, int);
int gen_alloc_dealloc(int, int, ITEM *);
void check_alloc_clauses(ITEM *, ITEM *, int *, int *);
void check_dealloc_clauses(ITEM *, ITEM *);
void gen_conditional_dealloc(int, int, int);
int gen_conditional_alloc(int, int, int);
void gen_conditional_dealloc_for_sym(int, int);
int gen_dealloc_for_sym(int, int);
void gen_automatic_reallocation(int, int, int);
void gen_dealloc_etmps(void);
void sem_set_storage_class(int);
void check_and_add_auto_dealloc_from_ast(int);
void check_and_add_auto_dealloc(int);
void add_auto_dealloc(int);
int enter_lexical_block(int);
void exit_lexical_block(int);
void dmp_doif(FILE *f);
LOGICAL not_in_forall(const char *);
LOGICAL cuda_enabled(const char *);
LOGICAL in_device_code(int);
void sem_err104(int, int, const char *);
void sem_err105(int);
VAR *gen_varref_var(int, DTYPE);
void sem_fini(void);
int gen_set_type(int dest_ast, int src_ast, int std, LOGICAL insert_before,
                 LOGICAL intrin_type);
int mk_set_type_call(int arg0, int arg1, LOGICAL intrin_type);
int has_init_value(SPTR);

/* semant.c */
void semant_init(int noparse);
int getMscall(void);
int getCref(void);
void build_typedef_init_tree(int sptr, int dtype);
int internal_proc_has_ident(int ident, int proc);
void fixup_reqgs_ident(int sptr);
int queue_type_param(int sptr, int dtype, int offset, int flag);
int get_kind_parm_by_name(char *np, int dtype);
int get_parm_by_number(int offset, int dtype);
int get_parm_by_name(char *np, int dtype);
int chk_kind_parm_expr(int ast, int dtype, int flag, int strict_flag);
int chk_len_parm_expr(int ast, int dtype, int flag);
int get_len_set_parm_by_name(char *np, int dtype, int *val);
int cmp_len_parms(int ast1, int ast2);
int defer_pt_decl(int dtype, int flag);
void put_default_kind_type_param(int dtype, int flag, int flag2);
void put_length_type_param(DTYPE dtype, int flag);
int get_len_parm_by_number(int num, int dtype, int flag);
int all_len_parms_assumed(int dtype);
LOGICAL put_kind_type_param(DTYPE dtype, int offset, int value, int expr,
                            int flag);
void llvm_set_tbp_dtype(int dtype);
int get_unl_poly_sym(int mem_dtype);
int has_type_parameter(int dtype);
int has_length_type_parameter_use(int dtype);
DTYPE create_parameterized_dt(DTYPE dtype, LOGICAL force);
DTYPE get_parameterized_dt(DTYPE dtype);
int is_parameter_context();
bool in_intrinsic_decl(void);
int get_entity_access();

/**
 * \brief Deferred procedure interface.
 */
typedef struct {
  SPTR iface;       /**< sptr of interface name */
  DTYPE dtype;      /**< dtype of TY_PROC data type record */
  SPTR proc;        /**< sptr of external/dummy procedure */
  SPTR mem;         /**< sptr of the procedure member/component */
  int lineno;       /**< line number of the statement */
  char *iface_name; /**< iface name string */
  int pass_class;   /**< set if pass arg has class set */
  char *tag_name;   /**< name of pass arg dtype tag */
  int sem_pass;     /**< semantic pass that this symbol was set */
  int stype;        /**< STYPE of iface */
  SPTR scope;       /**< scope of the procedure pointer declaration */
  SPTR proc_var;    /**< the procedure variable */
  int internal;     /**< value of gbl.internal when processing proc or mem */
} IFACE;

typedef struct ident_proc_list {
  char *proc_name; /* internal procedure name */
  int usecnt;      /* # times ident for this proc seen in contains proc */
  struct ident_proc_list *next;
} IDENT_PROC_LIST;

typedef struct ident_list {
  char *ident;                /* ident name seen in an internal procedure */
  IDENT_PROC_LIST *proc_list; /* list of internal proc names that use ident */
  struct ident_list *next;
} IDENT_LIST;

#define _INF_CLEN 500

/* program statement phase types:
 *
 *  INIT - nothing seen yet (initial value)
 *  HEADER - SUBROUTINE, FUNCTION, BLOCKDATA,
 *      PROGRAM, MODULE, SUBMODULE
 *  USE - USE statements seen
 *  IMPORT - IMPORT statements seen
 *  IMPLICIT - IMPLICIT statements
 *      PARAMETER may intersperse
 *  SPEC - Specification statements or
 *      statement function definitions
 *      PARAMETER, DATA, NAMELIST may intersperse
 *  EXEC - Executable statements
 *      DATA, NAMELIST may intersperse
 *  CONTAIN - CONTAINS statement
 *  INTERNAL - Internal/module subprograms
 *  END - END statement
 *  END_MODULE - END statement for a module (actual value is negative and is
 *               the minimum value)
 *
 *  NOTES:
 *     PARAMETER does not explicitly set
 *     pgphase unless pgphase is < IMPLICIT in
 *     which case it's set to IMPLICIT.
 *     DATA, NAMELIST do not explicitly set
 *     pgphase unless pgphase is < SPEC in which
 *     case it's set to SPEC.
 */
typedef enum {
  PHASE_END_MODULE = -1,
  PHASE_INIT = 0,
  PHASE_HEADER = 1,
  PHASE_USE = 2,
  PHASE_IMPORT = 3,
  PHASE_IMPLICIT = 4,
  PHASE_SPEC = 5,
  PHASE_EXEC = 6,
  PHASE_CONTAIN = 7,
  PHASE_INTERNAL = 8,
  PHASE_END = 9
} PHASE_TYPE;

/*  declare global semant variables:  */

typedef struct {
  int end_host_labno;      /* label number (not symbol table sptr) if the
                            * END statement of the host subprogram which
                            * contains internal subprogram is labeled.
                            * This 'label' is found when the end statement
                            * is processed during the first pass and is
                            * emitted when the host's CONTAINS statement
                            * is processed during the second pass.
                            */
  int doif_size;           /* size in records of DOIF stack area.  */
  DOIF *doif_base;         /* base pointer for DOIF stack area. */
  int doif_depth;          /* current DO-IF nesting level */
  SPTR index_sym_to_pop;   /* DO index symbol, pop off hash link at loop end */
  SPTR doconcurrent_symavl; /* stb.stg_avail value at do concurrent entry */
  DTYPE doconcurrent_dtype; /* explicit do concurrent index data type */
  int eqvlist;             /* head of list of equivalences */
  EQVV *eqv_base;          /* list of equivalences */
  int eqv_size;
  int eqv_avail;
  int *eqv_ss_base;        /* subscripts for equivalences */
  int eqv_ss_size;
  int eqv_ss_avail;
  int flabels;             /* pointer to list of ftn ref'd labels */
  int nml;                 /* pointer to list of namelist symbols */
  int funcval;             /* pointer to variable for function ret val */
  PHASE_TYPE pgphase;      /* statement type seen so far */
  int gdtype;              /* global data typ, a DT_ value */
  int ogdtype;             /* original global data type (i.e. before *n
                              modification), a DT_ value */
  int gty;                 /* global data type (i.e. before *n
                              modification), a TY_ value. */
  int gcvlen;              /* global character type size */
  int deferred_func_kind;  /* AST of unresolved func retval KIND expr */
  int deferred_func_len;   /* AST of unresolved func retval LEN expr */
  int deferred_dertype;    /* sptr of unresolved derived type func return */
  int deferred_kind_len_lineno; /* linenbr of unresolved func return type
                                   KIND/LEN */
  int atemps;              /* avail counter for array bounds temporaries */
  int itemps;              /* avail counter for temporaries named 'ixxx' */
  int ptemps;              /* avail counter for inliner ptr temporaries */
  bool savall;             /* top-level SAVE statement w.o. symbols specified */
  bool savloc;             /* possibly one or more local variables SAVE'd */
  bool autoloc;            /* at least one local AUTOMATIC variable */
  int none_implicit;       /* IMPLICIT NONE seen? - zero vs. nonzero */
  STSK *stsk_base;         /* base pointer for structure stack area */
  int stsk_size;           /* size in records of structure stack area */
  int stsk_depth;          /* current structure depth (i.e. stack top) */
  int stag_dtype;          /* structure tag dtype pointer */
  int psfunc;              /* next <var ref> may be lhs of statement func */
  LOGICAL dinit_error;     /* error flag during DATA stmt processing */
  int dinit_count;         /* # elements left in current dcl id to init */
  LOGICAL dinit_data;      /* TRUE if in DATA stmt, FALSE if type dcl or
                              structure init stmt */
  int dinit_nbr_inits;     /* number of ICT/IVL initialization pairs written
                              to the dinit file (astb.df) */
  LOGICAL ignore_stmt;     /* TRUE => parser is to ignore current stmt */
  int switch_size;         /* size of switch/CGOTO list area */
  int switch_avl;          /* next available word in switch list area */
  LOGICAL temps_reset;     /* TRUE if semant general temps can be resused */
  LOGICAL in_stfunc;       /* in statement function def */
  int in_dim;              /* in <dimension list> */
  int in_struct_constr;    /* 0 if false, else sptr of derived type tag */
  SCOPESTACK *scope_stack; /* pushed and popped as scopes are entered/left*/
  int scope_level;         /* starts at zero */
  int scope_size;          /* size of scope stack */
  int next_unnamed_scope;  /* index of next interface or parallel scope */
  int block_scope;         /* index of current innermost BLOCK scope */
  SPTR construct_sptr;     /* current innermost BLOCK or DO CONCURRENT
                              construct ST_BLOCK symbol */
  /* bounds and arrdim are filled in during semantic processing of <dim list>
   * specifiers, and processed by semutil2.c:mk_arrdsc() to create an array
   * descriptor (TY_ARRAY data type record). */
  struct _sem_bounds bounds[MAXDIMS];
  struct _sem_arrdim arrdim;
  int last_std;  /* last std created */
  int tkntyp;    /* token effecting semant reduction */
  SEQL seql;     /* records [NO]SEQUENCE:
                  *    type:
                  *        0   -- statement not seen
                  *        's' -- SEQUENCE
                  *        'n' -- NOSEQUENCE
                  *    next:  list of SEQL items, one for each variable
                  */
  int dtemps;    /* avail counter for 'd' temporaries */
  int interface; /* depth of interface blocks (0 => no interface) */
  INTERF *interf_base;
  int interf_size;
  argpos_t *argpos;             /* keyword arguments in positional order */
  DOSTACK dostack[MAX_DOSTACK]; /* stack for evaluating implied do's */
  DOSTACK *top;           /* next top of stack for evaluating implied do's */
  ITEM *p_dealloc;        /* pointer to list of dynamically allocated arrays,
                           * allocatable derived types, and derived types with
                           * allocatable components which must deallocated upon
                           * end of statement */
  ITEM *p_dealloc_delete; /* pointer to list of statements that
                           * can be deleted if the associated dynamically-
                           * allocated array isn't needed after all */
  int mod_cnt;            /* incremented if MODULE is seen */
  SPTR mod_sym;           /* ST_MODULE symbol for the MODULE subprogram */
  SPTR submod_sym;        /* original ST_MODULE symbol for SUBMODULE */
  int mod_public_level;   /* scope level of public USEs in module */
  int use_seen;           /* the current subprogram has a USE stmt */
  ACCL accl;              /* records PUBLIC/PRIVATE:
                           *    type:
                           *        0   -- 'default' access statement not seen
                           *        'u' -- 'default' is PUBLIC
                           *        'v' -- 'default' is PRIVATE
                           *    next:  list of ACCL items, one for each variable
                           */
  LOGICAL atomic[3]; /* atomic update: three element flag to record when the
                      * directive is seen (atomic[1]), whether or not atomic
                      * was the previous statement (atomic[0]), and whether
                      * or not endatomic needs to be generated (atomic[2])
                      */
  struct {           /* master/endmaster */
    int cnt;         /* counter */
    int lineno;      /* line number of master */
    int ast;         /* ast of master */
  } master;
  struct {      /* critical/endcritical */
    int cnt;    /* counter */
    int lineno; /* line number of critical */
    int ast;    /* ast of critical */
  } critical;
  ITEM *intent_list;       /* list of variables, not in an interface, for which
                            * INTENT was specfied */
  LOGICAL symmetric;       /* SYMMETRIC statement w.o. symbols specified */
  int which_pass;          /* which semantic analyzer pass - 0 or 1 */
  LOGICAL stfunc_error;    /* error occurred when referencing a stmt function
                            * while defining a statement function.
                            */
  LOGICAL mod_public_flag; /* when processing module contained routines,
                            * is the default public or private? */
  LOGICAL mod_dllexport;   /* Win64 dllexport seen, module symbols must be
                            * exported */
  SC_KIND sc;              /* SC_LOCAL or SC_PRIVATE for temporaries */
  int orph;                /* set wherever we see clause in orphan
                              and clause take private, shared
                               0  - not in
                               >0 - in
                           */
  int parallel;            /* parallel nesting level - PARALLEL, DOACROSS,
                            * PARALLELDO, PARALLELSECTIONS:
                            *  0 - not parallel
                            * >0 - parallel nesting level (1 => outermost)
                            */
  LOGICAL expect_do;       /* next statement after DOACROSS, PDO, or
                            * PARALLELDO needs to be a DO.
                            */
  int expect_acc_do;       /* next statement after ACC DO or ACC REGION DO
                            * needs to be a DO.
                            */
  int collapsed_acc_do;    /* value of collapse clause for acc loop */
  int seq_acc_do;    /* acc loop with 'seq' clause */
  int expect_cuf_do; /* next statement after CUF KERNELS needs to be a DO.  */
  LOGICAL close_pdo; /* A DO loop for a PDO, PARALLELDO, or DOACROSS
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
  LOGICAL expect_simd_do; /* next statement after SIMD construct
                           * to be a DO.
                           */
  LOGICAL expect_dist_do; /* next statement after SIMD construct
                           * to be a DO.
                           */
  int target;             /* use for OpenMP target */
  int teams;              /* use for OpenMP teams */

  struct { /* For atomic smp directive */
    int is_acc;
    int lineno;      /* line number of atomic */
    LOGICAL seen;    /* atomic directive just seen */
    LOGICAL pending; /* atomic directive not yet applied */
    LOGICAL apply;   /* to be applied */
    int accassignc;  /* assigment statement count*/
    int action_type; /* update|read|write|capture */
#define ATOMIC_UNDEF -1
#define ATOMIC_UPDATE 1
#define ATOMIC_READ 2
#define ATOMIC_WRITE 3
#define ATOMIC_CAPTURE 4
    int ast;       /* ast of generated A_MP_CRITICAL, or
                      genreated A_ACC_ATOMIC */
    int rmw_op;    /* AOP_ADD, AOP_SUB, etc */
    int mem_order; /* AOP_UNDEF: if this isn't read-modify-write */

  } mpaccatomic;
  LOGICAL is_hpf;     /* is this statement in !hpf$? */
  int hpfdcl;         /* available index for the hpf declarations
                       * whose semantic processing is deferred
                       * until the first executable is seen. The
                       * the hpf declarations are saved as indices
                       * [0, sem.hpfdcl-1].
                       */
  int ssa_area;       /* which getitem area to use for <ssa> */
  LOGICAL use_etmps;  /* flag to indicate that allocated temps created
                       * for terms in an expression need to be saved
                       * in the etmp list; they need to deallocated
                       * after the computation of the expression.
                       */
  ITEM *etmp_list;    /* list of the temps allocated for an
                       * expression.
                       */
  ITEM *auto_dealloc; /* list of allocatable arrays that need to be
                       * automatically deallocated (F95 feature).
                       */
  int blksymnum;
  LOGICAL ignore_default_none; /* don't perform the OMP DEFAULT(NONE) check */
  int collapse;                /* collapse value for the pardo or pdo */
  int collapse_depth;          /* depth of collapse loop; 1 => innermost */
  int task;                    /* depth of task
                                *  0 - not in task
                                * >0 - task nesting level (1 => outermost)
                                */
  int alloc_std;               /* std of ALLOCATE generated by
                                *  semutil2.c:gen_alloc_dealloc()
                                */
  struct {                     /* info of a call to an array function */
    int try
      ;               /* enable collection - when rhs of an assn */
    int sptr;         /* the function being called */
    int return_value; /* the ast of the temp which is the return
                       * value of the temp.
                       */
    int call_std;     /* the std of the call to the function */
    int alloc_std;    /* the std of the allocate of the temp if
                       * dynamic.
                       */
  } arrfn;
  LOGICAL in_enum;
  int *non_private_base; /* variables that cannot appear in a */
  int non_private_size;  /* private clause */
  int non_private_avail;
  int *typroc_base; /* TY_PROC dtypes created */
  int typroc_size;
  int typroc_avail;
  IFACE *iface_base;
  int iface_size;
  int iface_avail;
  LOGICAL class;            /* true if processing poly variable */
  int type_mode;            /* mode of type declaration:
                             * 0 - not within type
                             * 1 - within type
                             * 2 - within type and contains seen
                             */
  ITEM **tbp_arg;           /* saved type bound procedure argument stack */
  int tbp_arg_cnt;          /* tbp_arg stack depth */
  int tbp_access_stmt;      /* used to note private stmt after a
                             * contains statement within a type
                             * declaration (i.e., type_mode == 2)
                             * 0 - no stmt specified
                             * 1 - private specified
                             */
  int tbp_interface;        /* interface-name for deferred tbp processing*/
  int generic_tbp;          /* true if processing generic type bound proc */
  ITEM *auto_finalize;      /* list of objects that need to be finalized */
  int select_type_seen;     /* true if we just processed select type stmt */
  int param_offset;         /* counts # params for parameterized type */
  int kind_type_param;      /* currently processed  kind type parameter */
  int new_param_dt;         /* currently processed param derived type */
  ITEM *type_initialize;    /* list of parameterized type objects for init*/
  int extends;              /* type extension tag during type processing */
  int type_param_sptr;      /* currently processed type param sptr */
  int param_struct_constr;  /* true when process param struct constructor */
  int type_param_candidate; /* param offset for either len or kind */
  ITEM *len_candidate;      /* expression used for len */
  ITEM *kind_candidate;     /* expression used for kind */
  int len_type_param;       /* offset of param used for length */
  int param_assume_sz;      /* set when current type parameter is assume sz */
  int param_defer_len;      /* set when current type parameter is defer len */
  int defined_io_type;      /* set when we're processing defined IO stmts
                             * 1 = read(formatted), 2 = read(unformatted)
                             * 3 = write(formatted), 4 = write(unformatted)
                             */
  int defined_io_seen;      /* set when processing defined I/O item */
  struct {
    int allocs;
    int nodes;
  } stats;
  LOGICAL seen_import;        /* import stmt in an interface seen */
  void *save_aconst;          /* saves SST of array constructor */
  ITEM *alloc_mem_initialize; /* list of allocatable members to initialize */
  LOGICAL ieee_features;      /* USE ieee_features seen */
  LOGICAL io_stmt;            /* parsing an IO statement */
  LOGICAL seen_end_module;    /* seen end module statement */
  LOGICAL contiguous;         /* -Mcontiguous */
  SPTR modhost_proc;          /* ST_PROC of a module host routine containing an
                               * internal procedure (set on demand)
                               */
  SPTR modhost_entry;         /* ST_ENTRY of a module host routine containing an
                               * internal procedure (set on demand)
                               */
  bool module_procedure;      /* in instantiated MODULE PROCEDURE <id> def'n */
  int array_const_level;      /* increment at the beginning of the processing
                               * array constructor and decrement when it finishes.
                               */
  STD_RANGE *ac_std_range;    /* list of ranges that holds statements generated from
                               * implied-do loop in array constructor.
                               */
  STD_RECORD *elp_stack;      /* all <elp> met in array constructor. */
  bool parsing_operator;      /* true when we are parsing an ST_OPERATOR */
  bool equal_initializer;     /* true if we are parsing an assignment
                               * initializer (e.g., integer :: a = 100)
                               */
  bool proc_initializer;      /* true if we are initializing a pointer 
                               * with a procedure name.
                               */
} SEM;

extern SEM sem;

/*
 * NTYPE - number of basic types; this must include the NCHARACTER
 * type even though it may not be an available feature.
 */
#define NTYPE 23

extern INT cast_types[NTYPE][2][2];

#define IS_INTRINSIC(st) (st == ST_INTRIN || st == ST_GENERIC || st == ST_PD)

#define INSIDE_STRUCT (sem.stsk_depth != 0)

void CheckDecl(int);
#define DCLCHK(sptr)       \
  {                        \
    if (sem.none_implicit) \
      CheckDecl(sptr);     \
  }

#define DOCHK(sptr) \
  if (DOVARG(sptr)) { \
    if (sem.doconcurrent_symavl) \
      error(1053, ERR_Severe, gbl.lineno, "DO CONCURRENT", CNULL); \
    else \
      error(115, 2, gbl.lineno, SYMNAME(sptr), CNULL); \
  }

#define IN_MODULE (sem.mod_cnt && gbl.internal == 0)
#define IN_MODULE_SPEC (sem.mod_cnt && gbl.currsub == 0)

/*  declare external functions called only from within semant: */

/* main.c */
void end_contained(void);

/* semsmp.c */
LOGICAL use_opt_atomic(int);
int emit_epar(void);
int emit_etarget(void);
void is_dovar_sptr(int);
void clear_no_scope_sptr(void);
void add_no_scope_sptr(int, int, int);
void pop_accel_vars(void);
void handle_accdecl(int keyword);
void check_no_scope_sptr(void);
void parstuff_init(void);
int emit_bcs_ecs(int);
void end_parallel_clause(int);
void end_teams();
void end_target();
void add_assign_firstprivate(int, int);
void accel_end_dir(int, LOGICAL);
void add_non_private(int);
void mk_cuda_builtins(int *, int *, int);
int mk_cuda_typedef(char *);
int mk_mbr_ref(int, char *);
void set_parref_flag(int, int, int);
void set_parref_flag2(int, int, int);
int is_sptr_in_shared_list(SPTR);
void set_private_encl(int, int);
void set_private_taskflag(int);
int find_outer_sym(int);
void par_add_stblk_shvar(void);
int do_distbegin(DOINFO *, int, int);

/* semutil.c */
void check_derived_type_array_section(int);
int gen_poly_element_arg(int ast, SPTR sptr, int std);
int add_ptr_assign(int, int, int);
void gen_contig_check(int dest, int src, SPTR sdsc, int lineno, bool cs, int std);
int collapse_begin(DOINFO *);
int collapse_add(DOINFO *);
void link_parents(STSK *, int);
void link_members(STSK *, int);
int ref_object(int);
LOGICAL ast_isparam(int ast);
int mk_component_ast(int, int, int);
int chk_pointer_intent(int, int);
int any_pointer_source(int);
int chk_pointer_target(int, int);
int mod_type(int, int, int, int, int, int);
int getbase(int);
int do_index_addr(int);
int do_begin(DOINFO *);
void do_lastval(DOINFO *);
int do_parbegin(DOINFO *);
void do_end(DOINFO *);
int mkmember(int, int, int);
LOGICAL legal_labelvar(int);
void resolve_fwd_refs(void);
bool in_save_scope(SPTR);
DOINFO *get_doinfo(int);
LOGICAL is_protected(int);
void err_protected(int, const char *);
void set_assn(int);

/* semfin.c */
void semfin(void);
void ipa_semfin(void);
void semfin_free_memory(void);
void fix_class_args(int sptr);
void llvm_fix_args(int sptr);
void do_equiv(void);
void init_derived_type(SPTR, int, int);

/* semsym.c */
int sym_in_scope(int, OVCLASS, int *, int *, int);
void sem_import_sym(int);
int test_scope(int);
int declref(int, SYMTYPE, int);
void set_internref_stfunc(int, int*);
int declsym(int, SYMTYPE, LOGICAL);
int refsym(int, OVCLASS);
int refsym_inscope(int, OVCLASS);
void enforce_denorm(void);
int getocsym(int, OVCLASS, LOGICAL);
int declobject(int, SYMTYPE);
int newsym(int);
int ref_ident(int);
int ref_ident_inscope(int);
int ref_storage(int);
int ref_storage_inscope(int);
int ref_int_scalar(int);
int ref_based_object(int);
int ref_based_object_sc(int, SC_KIND);
int refocsym(int, OVCLASS);
int sym_skip_construct(int);
SPTR block_local_sym(SPTR);
int declsym_newscope(int, SYMTYPE, int);
int decl_private_sym(int);
int sem_check_scope(int, int);

/* semfunc.c */
int get_static_type_descriptor(int sptr);
int get_type_descr_arg(int func_sptr, int arg);
int get_type_descr_arg2(int func_sptr, int arg);
int sc_local_passbyvalue(int sptr, int func_sptr);
LOGICAL allocatable_member(int sptr);
LOGICAL in_kernel_region(void);
int get_tbp_argno(int sptr, int dty2);
int get_generic_member(int dtype, int sptr);
int get_generic_member2(int dtype, int sptr, int argcnt, int *argno);
int generic_tbp_has_pass_and_nopass(int dtype, int sptr);
int get_generic_tbp_pass_or_nopass(int dtype, int sptr, int flag);
int get_specific_member(int dtype, int sptr);
int get_implementation(int dtype, int sptr, int flag, int *memout);
int _selected_char_kind(int con);
/* end semfunc.c */

/* semfunc2.c */
void set_pass_objects(int, int);
int intrinsic_as_arg(int);
int ref_entry(int);
int select_gsame(int);
char *make_kwd_str(int);
char *make_keyword_str(int, int);
LOGICAL get_kwd_args(ITEM *, int, const char *);
LOGICAL evl_kwd_args(ITEM *, int, const char *);
LOGICAL sum_scatter_args(ITEM *, int);
LOGICAL check_arguments(int, int, ITEM *, char *);
LOGICAL chk_arguments(int, int, ITEM *, char *, int, int, int, int *);
LOGICAL ignore_tkr(int, int);
LOGICAL ignore_tkr_all(int);
int iface_intrinsic(int);
void defer_arg_chk(SPTR formal, SPTR actual, SPTR subprog,
                   cmp_interface_flags, int lineno, bool performChk);
/* end semfunc2.c */

/* semgnr.c */
void check_generic(int);
void init_intrinsic_opr(void);
void bind_intrinsic_opr(int, int);
int get_intrinsic_oprsym(int, int);
int get_intrinsic_opr(int, int);
int dtype_has_defined_io(int);
void check_defined_io(void);
void add_overload(int, int);
void copy_specifics(int fromsptr, int tosptr);

/* semant2.c */
int test_private_dtype(int dtype);

/* semant3.c */
void set_construct_name(int name);
int get_construct_name(void);
void check_doconcurrent(int doif);
int has_poly_mbr(int sptr, int flag);
void push_tbp_arg(ITEM *item);
ITEM *pop_tbp_arg(void);
void err307(const char *, int, int);
void gen_init_unl_poly_desc(int dest_sdsc_ast, int src_sdsc_ast, int std);

/* xref.c */
void xrefinit(void);
void xrefput(int symptr, int usage);
void xref(void);
/* end xref.c */

/** \brief Constants representing tasks for type bound procedure (tbp)
 *  processing.
 *
 *  These are used with the task argument in the queue_tbp() function.
 */
typedef enum tbpTasks {
  TBP_CLEAR_ERROR = -1,    /**< Clear all entries in queue after an error */
  TBP_CLEAR,               /**< Clear all entries after normal processing */
  TBP_CLEAR_STALE_RECORDS, /**< Clear tbp_queue entries with stale dtypes */
  TBP_ADD_SIMPLE,   /**< Add tbp after parsing simple tbp (e.g., procedure tbp;)
                     */
  TBP_ADD_TO_DTYPE, /**< Add tbps to derived type dtype records */
  TBP_COMPLETE_ENDMODULE, /**< Complete tbp ST_MEMBERs in derived type after
                               parsing ENDMODULE */
  TBP_ADD_INTERFACE,  /**< Add interface name to queue if user specified one */
  TBP_ADD_IMPL,       /**< Add binding name and implementation name to queue.
                           Occurs when we parse something like
                           procedure x => y (where x is the binding name and y is
                           the implementation name). */
  TBP_PASS,           /**< Specify explicit pass argument for tbp */
  TBP_COMPLETE_FIN,   /**< Complete tbp ST_MEMBERS after processing module
                           CONTAINS, etc. Called from semfin() */
  TBP_INHERIT,        /**< Copy inherited tbps from parent type to child type */
  TBP_NOPASS,         /**< Specify NOPASS attribute for tbp */
  TBP_NONOVERRIDABLE, /**< Specify NON_OVERRIDABLE attribute for tbp */
  TBP_PRIVATE,        /**< Specify PRIVATE attribute for tbp */
  TBP_PUBLIC,         /**< Specify PUBLIC attribute for tbp */
  TBP_STATUS,   /**< Check to see if we have tbps to add to a derived type */
  TBP_DEFERRED, /**< Specify DEFERRED attribute for tbp */
  TBP_IFACE,    /**< Specify an external routine via an explicit interface for
                     the tbp's implementation */
  TBP_COMPLETE_END,     /**< Complete tbp ST_MEMBERs after parsing ENDFUNCTION,
                             ENPROGRAM, ENSUBROUTINE outside the scope of a
                             module. */
  TBP_COMPLETE_ENDTYPE, /**< Complete tbp ST_MEMBERs after parsing ENDTYPE
                             outside the scope of a module. */
  TBP_CHECK_CHILD,      /**< Check validity of child tbp with parent tbp */
  TBP_CHECK_PRIVATE, /**< Check validity of private child tbp with parent tbp */
  TBP_CHECK_PUBLIC,  /**< Check validity of public child tbp with parent tbp */
  TBP_COMPLETE_GENERIC, /**< Complete tbp ST_MEMBERs for generic tbps. This
                             task is invoked in various places of generic and
                             operator processing. */
  TBP_ADD_FINAL,        /**< Add final subroutine to queue */
  TBP_FORCE_RESOLVE     /**< Force resolution of tbps in tbpQueue */

} tbpTask;

/* semtbp.c */
int queue_tbp(int sptr, int bind, int offset, int dtype, tbpTask task);
void ensure_no_stale_tbp_queue_entries(void);

/** \brief These are constants used by SST_DIMFLAG and A_MASK to represent
 *         empty subscripts (e.g., (:), (:,:), (:,:,:), etc. )
 */
typedef enum dimMask {
  lboundMask = 0x1, /**< empty lower bound mask */
  uboundMask = 0x2, /**< empty upper bound mask */
  strideMask = 0x4  /**< empty stride mask */
} dimMask;
