/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief FIXME
 */

void enter_local_mode(int std);            /* outconv.c */
void exit_local_mode(int std);             /* outconv.c */
void convert_forall(void);                 /* outconv.c */
int conv_forall(int std);                  /* outconv.c */
void forall_dependency_analyze(void);      /* outconv.c */
int gen_islocal_index(int, int, int, int); /* func.c */
/* commgen.c */
int gen_localize_index(int, int, int, int); /* commgen.c */
int inline_hlocalizebnds(int, int, int, int, int, int, int, int, int);
void generate_hlocalizebnds(int); /* commgen.c */
void outvalue_hlocalizebnds(int); /* commgen.c */
void generate_hcycliclp(int);     /* commgen.c */
void outvalue_hcycliclp(int);     /* commgen.c */
void put_forall_pcalls(int fstd); /* commopt.c */

/* func.c */
int convert_subscript(int);               /* func.c */
LOGICAL is_shift_conflict(int, int, int); /* func.c */
void check_pointer_type(int past, int tast, int stmt,
			LOGICAL is_sourced_allocation);
int search_conform_array(int ast, int flag);

LOGICAL contains_ast(int, int);       /* ast.c */
LOGICAL contains_call(int);           /* f90vect.c */
int rename_forall_list(int);          /* rest.c */
int concatenate_list(int, int);       /* vsub.c */
LOGICAL is_multiple_idx_in_list(int); /* vsub.c */
void unvisit_every_sptr(void);        /* dpm_out.c */
int sptr_of_subscript(int);           /* ast.c */
int sym_of_ast(int ast);              /* ast.c */
void init_gstbl(void);                /* commopt.c */
int get_gstbl(void);                  /* commopt.c */
void free_gstbl(void);                /* commopt.c */
void init_brtbl(void);                /* commopt.c */
int get_brtbl(void);                  /* commopt.c */
void free_brtbl(void);                /* commopt.c */
void init_rmtbl(void);                /* commopt.c */
int get_rmtbl(void);                  /* commopt.c */
void free_rmtbl(void);                /* commopt.c */
void reset_init_idx(void);
LOGICAL getbit(int, int);                             /* dpm_out.c */
LOGICAL is_vector_subscript(int, int);                /* vsub.c */
int process_forall(int);                              /* vsub.c */
LOGICAL is_no_comm(int, int);                         /* rest.c */
LOGICAL is_no_comm_expr(int, int);                    /* vsub.c */
LOGICAL is_no_rcomm_expr(int, int);                   /* vsub.c */
int insert_comm_before(int, int, LOGICAL *, LOGICAL); /* rest.c */
int localize_for_cyclic_expr_sub(int, int);           /* rest.c */
void copy_surrogate_to_bnds_vars(DTYPE, int, DTYPE, int, int);
void copy_desc_to_bnds_vars(int sptrdest, int desc, int memdesc, int std);
void emit_fl(void);                       /* dpm_out.c */
void init_sdsc_from_dtype(int sptr, DTYPE, int before_std); /* dpm_out.c */
void make_temp_descriptor(int, SPTR, SPTR, int); /* dpm_out.c */
int init_sdsc(int sptr, DTYPE dtype, int before_std, int parent_sptr);	/* semutil2.c */
void ipa_restore_dtb(char *line);         /* dpm_out.c */
void transform_call(int, int);
LOGICAL is_scalar(int a, int);
LOGICAL is_idx(int a, int);
void forall_init(void);
LOGICAL is_same_alnd(int, int);
LOGICAL is_same_secd(int, int);
int dist_symbol(int);

LOGICAL is_array_type(int);
void rt_outvalue(void);
int mk_ftb(void);
void init_ftb(void);
void init_bnd(void);
LOGICAL is_ordered(int, int, int, int[MAXDIMS], int *);
LOGICAL is_duplicate(int, int);
int delete_astli(int list, int);
void forall_lhs_indirection(int);
void forall_rhs_indirection(int);
LOGICAL scatter_class(int);
LOGICAL is_legal_rhs(int, int, int);
LOGICAL is_legal_lhs(int, int);
int is_scatter_subscript(int, int, int);
void search_forall_idx(int, int, int *, int *);
void process_rhs_sub(int);
void globalize_subscripts(int, int);
void globalize_expr(int);
void transform_wrapup(void);
void transform_ast(int, int);
int get_dist_type(int, int);
int dist_get(int);
int same_template(int, int);
int same_template_without_single(int, int);
LOGICAL is_replicated(int);
LOGICAL is_same_alignment(int, int, int, int);
LOGICAL is_same_single_alignment(int, int);
LOGICAL is_cyclic_dist(int sptr);
LOGICAL is_same_array_alignment(int, int);
LOGICAL is_same_dist_proc(int, int);
LOGICAL is_same_array_shape(int, int);
LOGICAL is_single_alignment(int align);
int gen_single(int sptr, int memberast);
int is_array_element_in_forall(int, int);
LOGICAL is_name_in_expr(int, int);
LOGICAL is_lonely_idx(int, int);
LOGICAL expr_dependent(int, int, int, int);   /* iterat.c */
LOGICAL subscr_dependent(int, int, int, int); /* iterat.c */
LOGICAL is_equivalence(int, int);
LOGICAL is_dist_all_star(int);
int process_lhs_sub(int std, int ast);
int tag_forall_comm(int);
void owner_computes_rule(int);
int replace_expr(int, int, int, int);
void replace_common_proc_template(void);
void guard_forall(int);
void delete_stmt(int);
void search_idx(int ast, int list, int *astli, int *base, int *stride);
void rewrite_calls(void);
int trans_getbound(int, int);
void rewrite_forall(void);
void rewrite_forall_pure(void);
LOGICAL is_dependent(int lhs, int rhs, int forall, int, int);
LOGICAL is_pointer_dependent(int sptr, int sptr1);
void emit_alnd_secd(int, int, LOGICAL, int, int);
int get_allobnds(int, int);
LOGICAL is_indirection_in_it(int);
LOGICAL is_nonscalar_indirection_in_it(int);
LOGICAL is_dist_array_in_expr(int ast);
LOGICAL is_vector_indirection_in_it(int, int);
int insert_maskend(int, int);
void original_align_or_dist(int, int *, int *);
LOGICAL is_kopy_in_needed(int);
void find_get_scalar(void);
LOGICAL is_use_lhs_final(int, int, LOGICAL, LOGICAL, int);
int make_sec_from_ast(int, int, int, int, int);
int make_sec_from_ast_chk(int, int, int, int, int, int);
int rewrite_interface_args(int func, int arg, int pos);
void get_dist_intr(int intr, int arg_list, int old_arg_list, int nargs);
void get_dist_func(int arg_list, int old_arg_list, int nargs);
LOGICAL needs_redim(int arg);
void init_dtb(void);
void free_dtb(void);
void init_fl(void);
void dpm_out_init(void);
void dpm_out_fe_gend(void);
void set_assumsz_bound(int arg, int entry);
void set_assumed_bounds(int arg, int entry, int actual);
int my_pe(void);                       /* shmem.c */
void extr_f77_local(int std, int ast); /* dpm_out.c */
LOGICAL pta_conflict(int ptrstdx, int ptrsptr, int targetstdx, int targetsptr,
                     int targetpointer, int targettarget);     /* pointsto.c */
int pta_target(int ptrstdx, int ptrsptr, int *ptag, int *pid); /* pointsto.c */
LOGICAL pta_aligned(int ptrstdx, int ptrsptr);                 /* pointsto.c */
bool pta_stride1(int ptrstdx, int ptrsptr);                 /* pointsto.c */

struct arg_gbl {
  int std;
  int lhs;
  LOGICAL used;
  LOGICAL inforall;
};
extern struct arg_gbl arg_gbl;

#define FORALL_AREA 10

struct forall_gbl {
  int s0, s1, s2, s3, s4, s5; /* std between block forall */
};
extern struct forall_gbl forall_gbl;

/* pre_processing_loop information */
struct pre_loop {
  int s0;    /* center position of loop */
  int count; /* sptr of the count */
  int inc;   /* count incremented */
  int dec;   /* count decremented */
};
extern struct pre_loop pre_loop;

extern int scope;

typedef struct {
  int ast;
  int type;
  int shape;
} FINFO;

typedef struct {
  FINFO *base;
  int avl;
  int size;
} FINFO_TBL;

extern FINFO_TBL finfot;

#define FINFO_AST(i) finfot.base[i].ast
#define FINFO_TYPE(i) finfot.base[i].type
#define FINFO_SHAPE(i) finfot.base[i].shape

#define CANONICAL_CAUSE 1
#define INTRINSIC_CAUSE 2
#define UGLYCOMM_CAUSE 3
#define DEPENDENCY_CAUSE 4
#define GETSCALAR_CAUSE 5
#define COPYSCALAR_CAUSE 6
#define COPYSECTION_CAUSE 7
#define MANYRUNTIME_CAUSE 8
#define PURECOMM_CAUSE 9
#define UGLYPURE_CAUSE 10
#define UGLYMASK_CAUSE 11
#define NOPARALLELISM_CAUSE 13

typedef struct itemlist {
  struct itemlist *next;
  int item;
  int nitem;
} LITEMF;

void plist(LITEMF *, int); /* commopt.c */
int glist(LITEMF *, int);  /* commopt.c */
LITEMF *clist(void);       /* commopt.c */

typedef struct {
  int f1;
  int f2;
  LITEMF *f3;
  int f4[7];
} TABLE;

struct tbl {
  TABLE *base;
  int avl;
  int size;
};

extern struct tbl tbl;
extern struct tbl pertbl;
extern struct tbl gstbl;
extern struct tbl brtbl;
extern struct tbl rmtbl;

struct pure_gbl {
  int local_mode;
  int end_critical_region;
  int end_master_region;
};

extern struct pure_gbl pure_gbl;
