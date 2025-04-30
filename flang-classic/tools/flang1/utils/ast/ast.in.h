/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief ast.h - AST definitions for Fortran
 */

/* clang-format off */
.ST

.SE

/* overloaded macros accessing shared fields */

#define A_STDG(s)   A_HSHLKG(s)
#define A_STDP(s,v) A_HSHLKP(s,v)
#define A_OPTYPEG(s)   A_hw21G(s)
#define A_OPTYPEP(s,v) A_hw21P(s,v)
#define A_TKNG(s)   A_hw21G(s)
#define A_TKNP(s,v) A_hw21P(s,v)
#define A_NTRIPLEG(s)   A_hw21G(s)
#define A_NTRIPLEP(s,v) A_hw21P(s,v)
#define A_SCHED_TYPEG(s)   A_hw21G(s)
#define A_SCHED_TYPEP(s,v) A_hw21P(s,v)
#define A_CANCELKINDG(s)   A_hw21G(s)
#define A_CANCELKINDP(s,v) A_hw21P(s,v)
#define A_ARGCNTG(s)   A_hw22G(s)
#define A_ARGCNTP(s,v) A_hw22P(s,v)
#define A_NCOUNTG(s)   A_hw22G(s)
#define A_NCOUNTP(s,v) A_hw22P(s,v)
#define A_ORDEREDG(s)   A_hw22G(s)
#define A_ORDEREDP(s,v) A_hw22P(s,v)
#define A_MEM_ORDERG(s)   A_hw22G(s)
#define A_MEM_ORDERP(s,v) A_hw22P(s,v)

/* A_BINOP, A_UNOP OP_ macros; also used by semant */

#define OP_NEG 0
#define OP_ADD 1
#define OP_SUB 2
#define OP_MUL 3
#define OP_DIV 4
#define OP_XTOI 5
#define OP_XTOX 6
#define OP_CMP 7
#define OP_AIF 8
#define OP_LD 9
#define OP_ST 10
#define OP_FUNC 11
#define OP_CON 12
#define OP_CAT 13
#define OP_LOG 14
#define OP_LEQV 15
#define OP_LNEQV 16
#define OP_LOR 17
#define OP_LAND 18
#define OP_EQ 19
#define OP_GE 20
#define OP_GT 21
#define OP_LE 22
#define OP_LT 23
#define OP_NE 24
#define OP_LNOT 25
/* remaining OP_ defines are not used by semant */
#define OP_LOC 26
#define OP_REF 27
#define OP_VAL 28
#define OP_SCAND 29
#define OP_SCALAR 30
#define OP_ARRAY 31
#define OP_DERIVED 32
#define OP_BYVAL 33

/* AST attributes: for fast AST checking -- astb.attr is a table indexed
 * by A_<type>
 */
.TA
#define A_ISLVAL(a) (astb.attr[a]&__A_LVAL)
#define A_ISEXPR(a) (astb.attr[a]&__A_EXPR)

/* miscellaneous macros */

#define COMSTR(s) (astb.comstr.stg_base + A_COMPTRG(s))

/*=================================================================*/

/* ----- Auxiliary Structures ----- */

/* AST List Item (ASTLI) */

typedef struct {
    int      h1;
    int      h2;
    int      flags;
    int      next;
} ASTLI;

#define ASTLI_HEAD astb.astli.stg_base[0].next

#define ASTLI_SPTR(i) astb.astli.stg_base[i].h1
#define ASTLI_AST(i) astb.astli.stg_base[i].h1

#define ASTLI_PT(i) astb.astli.stg_base[i].h2
#define ASTLI_TRIPLE(i) astb.astli.stg_base[i].h2

#define ASTLI_FLAGS(i) astb.astli.stg_base[i].flags
#define ASTLI_NEXT(i) astb.astli.stg_base[i].next

/* ARG Table */

#define ARGT_CNT(i)  astb.argt.stg_base[i]
#define ARGT_ARG(i,j) astb.argt.stg_base[(i)+((j)+1)]

/* Array Subscript Descriptor (ASD) */

typedef struct {
    int    ndim;	/* number of dimensions for this descriptor */
    int    next;	/* next ASD with the same number of subscripts */
    int    subs[1];	/* 1 <= size <= 7; 0 <= index <= 6 */
} ASD;

#define ASD_NDIM(i) ((ASD *)&astb.asd.stg_base[i])->ndim
#define ASD_NEXT(i) ((ASD *)&astb.asd.stg_base[i])->next
#define ASD_SUBS(i,j) ((ASD *)&astb.asd.stg_base[i])->subs[j]

/* Shape Descriptor (SHD) */

typedef struct {
    /* A shape descscriptor is composed of 'n+1' elements to represent
     * a shape of rank 'n'.  The first element of a SHD is:
     *     lwb    -> ndim, rank of the shape descriptor
     *     upb    -> next, next SHD with the same number of subscripts
     *     stride -> not used
     * The ensuing 'n' elements describe the lower bound, upper bound, and
     * stride for each dimension.
     */
    int    lwb;	/* ast of lower bound */
    int    upb;	/* ast of upper bound */
    int    stride;	/* ast of stride */
} SHD;

#define SHD_NDIM(i) astb.shd.stg_base[i].lwb
#define SHD_NEXT(i) astb.shd.stg_base[i].upb
#define SHD_FILL(i) astb.shd.stg_base[i].stride
#define SHD_LWB(i,j) astb.shd.stg_base[i+j+1].lwb
#define SHD_UPB(i,j) astb.shd.stg_base[i+j+1].upb
#define SHD_STRIDE(i,j) astb.shd.stg_base[i+j+1].stride


/* Statement Descriptor (STD) */

typedef struct {
    int    ast;
    int    next;
    int    prev;
    int    label;
    int    lineno;
    int    findex;
    int    fg;		/* defined & referenced only by comm_post() */
    SPTR   blksym;      /* do concurrent body block sym */
#ifdef PGF90
    int    tag;		/* used for PFO */
    int    pta;		/* pointer target information */
    int    ptasgn;	/* pointer target pseudo-assignments */
    int    astd;	/* std of paired set (e.g. allocate/deallocate) */
    int    visit;	/* mark std after it is hoisted */
#endif
    union {
	int  all;
	struct {
	    unsigned  ex:1;
	    unsigned  st:1;
	    unsigned  br:1;
	    unsigned  delete:1;
	    unsigned  ignore:1;	 /* used by hl vectorizer */
	    unsigned  split:1;	 /* split the loop here */
	    unsigned  minfo:1;	 /* stmt has 'minfo' */
	    unsigned  local:1;	 /* stmt will not cause communication */
	    unsigned  pure:1;	 /* stmt references a PURE subprogram */
	    unsigned  par:1;	 /* stmt belongs to a parallel region */
	    unsigned  cs:1;	 /* stmt belongs to a critical section */
	    unsigned  parsect:1; /* stmt belongs to a parallel section */
	    unsigned  orig:1;	 /* stmt was original user statement */
	    unsigned  mark:1;
	    unsigned  task:1;    /* stmt belongs to a task */
	    unsigned  accel:1;   /* stmt belongs to an accelerator region */
	    unsigned  kernel:1;  /* stmt belongs to an cuda kernel */
	    unsigned  atomic:1;  /* stmt belongs to an atomic */
	    unsigned  ztrip:1;   /* stmt marked for array assignment */
	    unsigned  rescope:1; /* stmt marked for kernels rescope */
	    unsigned  indivisible:1; /* stmt is in indivisible structure(s) */
	}  bits;
    }  flags;
} STD;

#define STD_AST(i)     astb.std.stg_base[i].ast
#define STD_NEXT(i)    astb.std.stg_base[i].next
#define STD_PREV(i)    astb.std.stg_base[i].prev
#define STD_LABEL(i)   astb.std.stg_base[i].label
#define STD_LINENO(i)  astb.std.stg_base[i].lineno
#define STD_FINDEX(i)  astb.std.stg_base[i].findex
#define STD_FG(i)      astb.std.stg_base[i].fg
#define STD_BLKSYM(i)  astb.std.stg_base[i].blksym
#define STD_FIRST      astb.std.stg_base[0].next
#define STD_LAST       astb.std.stg_base[0].prev
#ifdef PGF90
#define STD_TAG(i)     astb.std.stg_base[i].tag
#define STD_PTA(i)     astb.std.stg_base[i].pta
#define STD_PTASGN(i)  astb.std.stg_base[i].ptasgn
#define STD_HSTBLE(i)  astb.std.stg_base[i].astd
#define STD_VISIT(i)   astb.std.stg_base[i].visit
#endif
#define STD_FLAGS(i)   astb.std.stg_base[i].flags.all
#define STD_EX(i)      astb.std.stg_base[i].flags.bits.ex
#define STD_ST(i)      astb.std.stg_base[i].flags.bits.st
#define STD_BR(i)      astb.std.stg_base[i].flags.bits.br
#define STD_DELETE(i)  astb.std.stg_base[i].flags.bits.delete
#define STD_IGNORE(i)  astb.std.stg_base[i].flags.bits.ignore
#define STD_SPLIT(i)   astb.std.stg_base[i].flags.bits.split
#define STD_MOVED(i)   STD_SPLIT(i)
#define STD_MINFO(i)   astb.std.stg_base[i].flags.bits.minfo
#define STD_LOCAL(i)   astb.std.stg_base[i].flags.bits.local
#define STD_PURE(i)    astb.std.stg_base[i].flags.bits.pure
#define STD_PAR(i)     astb.std.stg_base[i].flags.bits.par
#define STD_CS(i)      astb.std.stg_base[i].flags.bits.cs
#define STD_PARSECT(i) astb.std.stg_base[i].flags.bits.parsect
#define STD_ORIG(i)    astb.std.stg_base[i].flags.bits.orig
#define STD_MARK(i)    astb.std.stg_base[i].flags.bits.mark
#define STD_TASK(i)    astb.std.stg_base[i].flags.bits.task
#define STD_ACCEL(i)   astb.std.stg_base[i].flags.bits.accel
#define STD_KERNEL(i)  astb.std.stg_base[i].flags.bits.kernel
#define STD_ATOMIC(i)  astb.std.stg_base[i].flags.bits.atomic
#define STD_ZTRIP(i)   astb.std.stg_base[i].flags.bits.ztrip
#define STD_RESCOPE(i)   astb.std.stg_base[i].flags.bits.rescope
#define STD_INDIVISIBLE(i)   astb.std.stg_base[i].flags.bits.indivisible


/*=================================================================*/

/* hash table stuff */
#define HSHSZ 503

/* limits */
#define MAXAST   67108864


/*  AST typedef declarations:  */

typedef struct {
    int16_t type;
    uint8_t f1:1, f2:1, f3:1, f4:1, f5:1, f6:1, f7:1, f8:1;
    uint8_t hw1:8;
    int   hshlk;
    int   w3;
    int   w4;
    int   w5;
    int   w6;
    int   w7;
    int   w8;
    int   w9;
    int   w10;
    uint16_t hw21;
    uint16_t hw22;
    int   w12;
    int   opt1;
    int   opt2;
    int   repl;
    int   visit;
    int   shape;
    int   w18;
    int   w19;
}               AST;

/*   AST data declaration:  */

typedef struct {
    const char *atypes[AST_MAX + 1];
    int     attr[AST_MAX + 1];
    int     hshtb[HSHSZ + 1];
    STG_MEMBERS(AST);
    int     firstuast;
    int     i0;		/* 'predefined' ast for integer 0 */
    int     i1;		/* 'predefined' ast for integer 1 */
    int     k0;		/* 'predefined' ast for INT8 0 */
    int     k1;		/* 'predefined' ast for INT8 1 */
    int     ptr0;	/* 'predefined' ast for (void *)0 */
    int     ptr1;	/* 'predefined' ast for (void *)1 */
    int     ptr0c;	/* 'predefined' ast for non-present character I/O spec*/
    struct {
	STG_MEMBERS(int);
	int     hash[7];  /* max # of dimensions */
    } asd;
    STG_DECLARE(std, STD);
    STG_DECLARE(astli, ASTLI);
    STG_DECLARE(argt, int);
    struct {
	STG_MEMBERS(SHD);
	int     hash[7];  /* max # of dimensions */
    } shd;
    STG_DECLARE(comstr, char);
    UINT16      implicit[55];  /* implicit dtypes:
				* [ 0-25] a-z    [26-51] A-Z
				*    [52] $         [53] _     [55] none
				*/
    FILE   *astfil;	/* file pointer for (temporary) ast source file */
    FILE   *df;		/* temp file for saving data initializations */
    /*
     * the following members are initialized to values which reflect the
     * default type for the extents and subscripts of arrays.  The type could
     * either be 32-int or 64-bit (BIGOBJects & -Mlarge_arrays).
     *
     */
    struct {
        int     zero;	/* 'predefined' ast for ISZ_T 0 */
	int     one;	/* 'predefined' ast for ISZ_T 1 */
	int     dtype;	/* dtype used for the bound temps */
    } bnd;
}  ASTB;

extern ASTB  astb;
extern int   intast_sym[];

/** \brief Type of function passed to ast_traverse() and ast_traverse_all() */
typedef LOGICAL (*ast_preorder_fn) ();

/** \brief Type of function passed to ast_visit(), ast_traverse() and ast_traverse_all() */
typedef void (*ast_visit_fn) ();

/*   declare external functions from ast.c:  */

void ast_init (void);
void ast_fini (void);
int new_node (int);
int mk_id (int);
int mk_id_noshape (int);
int mk_init (int, DTYPE);
int mk_cnst (int);
int mk_cval (INT, DTYPE);
int mk_isz_cval (ISZ_T, DTYPE);
int mk_fake_iostat (void);
int mk_cval1 (INT, DTYPE);
void mk_alias (int, int);
int mk_label (int);
int mk_binop (int, int, int, DTYPE);
int mk_unop (int, int, DTYPE);
int mk_cmplxc (int, int, DTYPE);
int mk_paren (int, DTYPE);
int mk_convert (int, DTYPE);
int convert_int (int, DTYPE);
int mk_promote_scalar (int, DTYPE, int);
int mk_subscr (int, int *, int, DTYPE);
int mk_subscr_copy (int, int, DTYPE);
int mk_asd(int *, int);
int mk_triple (int, int, int);
int mk_substr (int, int, int, DTYPE);
int complex_alias(int);
int mk_member (int, int, DTYPE);
int mkshape (DTYPE);
int mk_mem_ptr_shape (int, int, DTYPE);
int mk_shape (void);
int mk_atomic(int, int, int, DTYPE);
int reduc_shape (int, int, int);
int increase_shape (int, int, int, int);
void add_shape_rank (int);
void add_shape_spec (int, int, int);
LOGICAL conform_shape (int, int);
int extent_of_shape (int, int);
int lbound_of_shape (int, int);
int ubound_of_shape (int, int);
int rank_of_ast (int);
int mk_zbase_expr (ADSC *);
int mk_mlpyr_expr (int, int, int);
int mk_extent_expr (int, int);
int mk_extent (int, int, int);
int mk_offset (int, int);
int mk_shared_extent (int, int, int);
int sym_of_ast (int);
int memsym_of_ast (int);
int procsym_of_ast (int);
LOGICAL pure_func_call(int);
LOGICAL elemental_func_call(int);
int sptr_of_subscript (int);
int left_array_symbol (int);
int left_subscript_ast (int);
int left_nonscalar_subscript_ast (int);
int dist_symbol (int);
int dist_ast (int);
LOGICAL is_whole_dim(int arr_ast, int i);
LOGICAL is_whole_array(int arr_ast);
LOGICAL simply_contiguous(int arr_ast);
LOGICAL bnds_remap_list(int subscr_ast);
int replace_ast_subtree (int, int, int);
int elem_size_of_ast (int);
int size_of_ast (int);
int mk_bnd_ast (void);
int mk_shared_bnd_ast (int);
int mk_stmt (int, DTYPE);
int mk_std (int);
int add_stmt (int);
int add_stmt_after (int, int);
int add_stmt_before (int, int);
void insert_stmt_after(int std, int stdafter);
void insert_stmt_before(int std, int stdbefore);
void remove_stmt(int std);
void move_stmt_before(int std, int stdbefore);
void move_stmt_after(int std, int stdafter);
void move_stmts_before(int std, int stdbefore);
void move_stmts_after(int std, int stdafter);
void move_range_before(int sstd, int estd, int stdbefore);
void move_range_after(int sstd, int estd, int stdafter);
void ast_to_comment (int);
int mk_comstr (char *str);
int mk_argt (int cnt);
void unmk_argt (int cnt);
void start_astli (void);
int add_astli (void);
void ast_implicit (int, int, DTYPE);
int begin_call (int, int, int);
void add_arg (int);
void finish_args (DTYPE, LOGICAL);
int mk_func_node (int, int, int, int);
int mk_assn_stmt (int, int, DTYPE);
LOGICAL contains_ast (int, int);
void ast_visit (int, int);
void ast_replace (int, int);
void ast_unvisit (void);
void ast_unvisit_norepl (void);
void ast_revisit (ast_visit_fn, int *);
int ast_rewrite (int);
void ast_clear_repl (int);
void ast_traverse (int, ast_preorder_fn, ast_visit_fn, int *);
void ast_traverse_all (int, ast_preorder_fn, ast_visit_fn, int *);
void ast_traverse_more (int, int *);
void ast_trav_recurse (int, int *);
void dump_one_ast (int);
void _dump_one_ast(int, FILE *);
void dump_ast_tree (int);
void dump_ast (void);
void dump_astli(int astli);
void dump_std (void);
void _dump_std (int, FILE *);
void dump_shape(int);
void _dump_shape(int, FILE *);
void dump_stg_stat (const char *);
void dbg_print_ast(int ast, FILE *fil); /* astout.c */

int ast_intr(int i_intr, DTYPE dtype, int cnt, ...);
int mk_default_int (int);
int mk_bnd_int (int);
int mk_smallest_val (DTYPE);
int mk_largest_val (DTYPE);
int mk_merge(int, int, int, DTYPE);
int get_atemp (void);
void set_atemp (int);
void delete_stmt (int);
int add_nullify_ast(int);
void printast(int);
int pass_sym_of_ast(int);
void end_param(void); /* astout.c */
void add_param(int); /*astout.c */
void astout_init(void); /* astout.c */
void put_memsym_of_ast(int ast, int sptr);
int replace_memsym_of_ast(int ast, SPTR sptr);
int has_assumshp_expr(int ast);
int has_adjustable_expr(int ast);
int has_pointer_expr(int ast);
int has_allocatable_expr(int ast);
int is_iso_cloc(int ast);
int is_iso_c_loc(int ast);
int is_iso_c_funloc(int ast);
int find_pointer_variable(int ast);
void find_pointer_target(int ast, int *pbase, int *psym);
void holtonum(char *cp, INT *num, int bc);
INT cngcon(INT oldval, int oldtyp, int newtyp);
INT negate_const(INT conval, DTYPE dtype);
INT const_fold(int opr, INT conval1, INT conval2, DTYPE dtype);
int resolve_ast_alias(int ast);
LOGICAL is_data_ast(int ast);
LOGICAL is_variable_ast(int ast);
LOGICAL is_array_ast(int ast);
LOGICAL has_vector_subscript_ast(int ast);
int get_ast_sptr(int ast);
int get_ast_asd(int ast);
DTYPE get_ast_dtype(int ast);
int get_ast_rank(int ast);
int rewrite_ast_with_new_dtype(int ast, DTYPE dtype);
int mk_duplicate_ast(int ast);
int get_ast_extents(int extent_asts[], int from_ast, DTYPE arr_dtype);
int get_ast_bounds(int lower_bound_asts[], int upper_bound_asts[],
                   int from_ast, DTYPE arr_dtype);
int add_extent_subscripts(int to_ast, int rank, const int extent_asts[], DTYPE elt_dtype);
int add_bounds_subscripts(int to_ast, int rank, const int lower_bound_asts[],
                          const int upper_bound_asts[], DTYPE elt_dtype);
int add_shapely_subscripts(int to_ast, int from_ast, DTYPE arr_dtype, DTYPE elt_dtype);
LOGICAL ast_is_sym(int ast);
SPTR get_whole_array_sym(int arr_ast);
