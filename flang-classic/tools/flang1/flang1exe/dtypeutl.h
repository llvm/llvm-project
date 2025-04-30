/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Fortran data type utility functions.
 */

extern int ty_to_lib[];

const char *target_name(DTYPE dtype);
int target_kind(DTYPE dtype);
ISZ_T size_of(DTYPE dtype);
int string_length(DTYPE dtype);
LOGICAL is_empty_typedef(DTYPE dtype);
LOGICAL is_zero_size_typedef(DTYPE dtype);
LOGICAL no_data_components(DTYPE dtype);
ISZ_T size_of_var(int sptr);
INT size_ast(int sptr, DTYPE dtype);
INT size_ast_of(int ast, DTYPE dtype);
INT string_expr_length(int ast);
DTYPE adjust_ch_length(DTYPE dtype, int ast);
DTYPE fix_dtype(int sptr, DTYPE dtype);
ISZ_T extent_of(DTYPE dtype);
DTYPE dtype_with_shape(DTYPE dtype, int shape);
ISZ_T ad_val_of(int sym);
int get_bnd_con(ISZ_T v);
int alignment(DTYPE dtype);
int alignment_of_var(int sptr);
int bits_in(DTYPE dtype);
void init_chartab(void);
void fini_chartab(void);
DTYPE get_type(int n, TY_KIND v1, int v2);
LOGICAL cmpat_func(DTYPE d1, DTYPE d2);
LOGICAL tk_match_arg(int formal_dt, int actual_dt, LOGICAL flag);
LOGICAL extends_type(int tg1, int tg2);
LOGICAL eq_dtype2(DTYPE d1, DTYPE d2, LOGICAL flag);
LOGICAL eq_dtype(DTYPE d1, DTYPE d2);
LOGICAL same_ancestor(DTYPE dtype1, DTYPE dtype2);
LOGICAL has_recursive_component(SPTR sptr);
LOGICAL has_finalized_component(SPTR sptr);
LOGICAL has_impure_finalizer(SPTR sptr);
LOGICAL has_layout_desc(SPTR sptr);
LOGICAL is_or_has_poly(SPTR sptr);
LOGICAL is_or_has_derived_allo(SPTR sptr);
LOGICAL cmpat_dtype(DTYPE d1, DTYPE d2);
LOGICAL cmpat_dtype_with_size(DTYPE d1, DTYPE d2);
LOGICAL cmpat_dtype_array_cast(DTYPE d1, DTYPE d2);
LOGICAL same_dtype(DTYPE d1, DTYPE d2);
void getast(int ast, char *string);
void getdtype(DTYPE dtype, char *ptr);
void dmp_dtype(void);
int dlen(int ty);
int _dmp_dent(DTYPE dtypeind, FILE *outfile);
int dmp_dent(DTYPE dtypeind);
void pr_dent(DTYPE dt, FILE *f);
int scale_of(DTYPE dtype, INT *size);
int fval_of(DTYPE dtype);
int kanji_len(const unsigned char *p, int len);
int kanji_char(unsigned char *p, int len, int *bytes);
int kanji_prefix(unsigned char *p, int newlen, int len);
DTYPE get_array_dtype(int numdim, DTYPE eltype);
void get_aux_arrdsc(DTYPE dtype, int numdim);
DTYPE dup_array_dtype(DTYPE o_dt);
DTYPE reduc_rank_dtype(DTYPE o_dt, DTYPE elem_dt, int astdim, int after);
int rank_of(DTYPE dtype);
int rank_of_sym(int sptr);
int lbound_of(DTYPE dtype, int dim);
int lbound_of_sym(int sptr, int dim);
int ubound_of(DTYPE dtype, int dim);
int ubound_of_sym(int sptr, int dim);
LOGICAL conformable(DTYPE d1, DTYPE d2);
int dtype_to_arg(DTYPE dtype);
int kind_of(DTYPE d1);
LOGICAL same_type_different_kind(DTYPE d1, DTYPE d2);
LOGICAL different_type_same_kind(DTYPE d1, DTYPE d2);
LOGICAL has_tbp(DTYPE dtype);
LOGICAL has_tbp_or_final(DTYPE dtype);
int chk_kind_parm_set_expr(int ast, DTYPE dtype);
void chkstruct(DTYPE dtype);
DTYPE is_iso_cptr(DTYPE d_dtype);
LOGICAL is_iso_c_ptr(DTYPE d_dtype);
LOGICAL is_iso_c_funptr(DTYPE d_dtype);
LOGICAL is_cuf_c_devptr(DTYPE d_dtype);
DTYPE get_iso_ptrtype(const char *name);
DTYPE get_iso_c_ptr(void);
LOGICAL is_array_dtype(DTYPE dtype);
DTYPE array_element_dtype(DTYPE dtype);
LOGICAL is_dtype_runtime_length_char(DTYPE dtype);
LOGICAL is_dtype_unlimited_polymorphic(DTYPE dtype);
LOGICAL is_procedure_ptr_dtype(DTYPE dtype);
DTYPE proc_ptr_result_dtype(DTYPE dtype);
void set_proc_ptr_result_dtype(DTYPE ptr_dtype, DTYPE result_dtype);
void set_proc_ptr_param_count_dtype(DTYPE ptr_dtype, int param_count);
LOGICAL is_procedure_dtype(DTYPE dtype);
void set_proc_result_dtype(DTYPE proc_dtype, DTYPE result_dtype);
void set_proc_param_count_dtype(DTYPE proc_dtype, int param_count);
SPTR get_struct_tag_sptr(DTYPE dtype);
SPTR get_struct_members(DTYPE dtype);
SPTR get_struct_initialization_tree(DTYPE dtype);
LOGICAL is_unresolved_parameterized_dtype(DTYPE dtype);
DTYPE change_assumed_char_to_deferred(DTYPE);
bool is_deferlenchar_ast(int);
bool is_deferlenchar_dtype(DTYPE);
