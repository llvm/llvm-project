/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief - Symbol utilities
 */

#ifndef SYMUTL_H_
#define SYMUTL_H_

#define SLIST_AREA 11

#define BLOCK_ENTRY_STD(sptr) (LABSTDG(STARTLABG(ENCLFUNCG(sptr))))
#define BLOCK_ENDPROLOG_STD(sptr) (ENTSTDG(ENCLFUNCG(sptr)))
#define BLOCK_EXIT_STD(sptr)  (LABSTDG(ENDLABG(ENCLFUNCG(sptr))))

int get_next_sym(const char *, const char *);
SPTR get_symtype(SYMTYPE, SPTR);
int get_next_user_sym(char *, char *);
int sym_get_scalar(const char *, const char *, int);
int sym_get_ptr(int);
int sym_get_ptr_name(char *);
int sym_get_offset(int);
int sym_get_array(const char *, const char *, int, int);
int sym_mkfunc(const char *, int);
int sym_mkfunc_nodesc(const char *, int);
int sym_mkfunc_nodesc_nocomm(const char *, int);
int sym_mkfunc_nodesc_expst(const char *nmptr, int dtype);
int sym_mknproc(void);
int sym_get_tmplate(int);
int sym_get_sec(const char *, int);
int sym_get_cp(void);
int sym_get_xfer(void);
int sym_get_arg_sec(int);
int sym_get_formal(int);
int get_temp_forall(int, int, int, int, int, int);
int get_temp_copy_section(int, int, int, int, int, int);
int get_temp_pure_replicated(int, int, int, int);
int mk_assign_sptr(int, const char *, int *, int, int *);
int chk_assign_sptr(int, const char *, int *, int, int, int *);
int mk_shape_sptr(int, int *, int);
int chk_reduc_sptr(int, const char *, int *, int, int, int, int *);
int mk_spread_sptr(int, const char *, int *, int, int, int, int, int *);
int mk_matmul_sptr(int, int, const char *, int *, int, int *);
int mk_transpose_sptr(int, const char *, int *, int, int *);
int mk_pack_sptr(int, int);
int mk_maxloc_sptr(int, int);
int search_forall_var(int, int);
int other_forall_var(int, int, int);
int mk_forall_sptr(int, int, int *, int);
int mk_forall_sptr_copy_section(int, int, int, int *, int);
int mk_forall_sptr_pure(int, int, int, int *, int);
int mk_mem_allocate(int, int *, int, int);
int mk_mem_deallocate(int, int);
int sym_get_global_array(void);
int check_member(int, int);
void dump_alignment(FILE *, int, int);
void dump_distribution(FILE *, int, int);
int first_hash(int sptr);
int find_dummy(int, int);
int find_array(int ast, int *returnast);
LOGICAL has_allocattr(int);
char *mangle_name_dt(const char *, const char *, int);
void fixup_srcalloc_bounds(int *, int *, int);

void check_alloc_ptr_type(int, int, DTYPE, int, int, int, int); /* func.c */
LOGICAL contiguous_section(int arr_ast);                        /* func.c */
int rewrite_lbound_ubound(int func_ast, int actual, int nextstd); /* func.c */

int gen_set_len_ast(int ast, SPTR ddesc, int sz); /* outconv.c */
LOGICAL inline_RTE_set_type(int, int, int, int, DTYPE, int); /* outconv.c */

int get_forall_subscr(int, int, int *, int);

void set_symutl_sc(int);
int get_next_sym_dt(const char *, const char *, int);
void trans_mkdescr(int);
int first_element(int ast);
int mk_forall_sptr_gatherx(int forall_ast, int lhs, int rhs, int *subscr,
                           int elem_dty);
int get_temp_pure(int forall_ast, int lhs, int rhs, int alloc_stmt,
                  int dealloc_stmt, int ast_dty);
void check_small_allocatable(int sptr);
LOGICAL was_implicit(int sptr);             /* symtab.c */
LOGICAL is_argp_in_entry(int ent, int arg); /* symtab.c */
int get_next_hash_link(int sptr, int task);
int findByNameStypeScope(char *symname, int stype, int scope);
LOGICAL is_array_sptr(int sptr);
LOGICAL is_unl_poly(int sptr);
bool is_impure(SPTR sptr);
LOGICAL needs_descriptor(int sptr);
bool proc_arg_needs_proc_desc(SPTR symfunc);
int find_descriptor_ast(int sptr, int ast);
SPTR get_member_descriptor(SPTR sptr);
int find_member_descriptor(int sptr);
int find_dummy_position(int proc_sptr, int arg_sptr);
int max_binding_invobj(int impl_sptr, int invobj);
LOGICAL is_tbp(int sptr);
LOGICAL is_final_procedure(int sptr);
LOGICAL is_tbp_or_final(int sptr);
int get_tmp_descr(DTYPE dtype);
int get_descriptor_length_ast(int descriptor_ast);
int symbol_descriptor_length_ast(SPTR sptr, int ast);
int get_value_length_ast(DTYPE value_dtype, int value_ast,
                         SPTR sptr, DTYPE sptr_dtype, int value_descr_ast);

#if DEBUG
void ds(int);
void dss(int, int);
void dsym(int);
void dsyms(int, int);
#endif

typedef struct {
  int sc;            /* storage class used for symutl-created scalars */
  int none_implicit; /* Copy of sem.none_implicit */
} SYMUTL;

extern SYMUTL symutl; /**< FIXME: what is this? */

void add_auto_len(int sym, int Lbegin);

#endif /* SYMUTL_H_ */
