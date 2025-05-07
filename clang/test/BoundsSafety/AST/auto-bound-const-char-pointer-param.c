
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -ast-dump -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
#include <stddef.h>
#include <stdint.h>

/*
 * The cases where __null_terminated should be added:
 */

// CHECK: ParmVarDecl {{.+}} const_char_parm 'const char *__single __terminated_by(0)'
void const_char_parm(const char *const_char_parm);

// CHECK: ParmVarDecl {{.+}} const_wchar_t_param 'const wchar_t *__single __terminated_by(0)'
void const_wchar_t_param(const wchar_t *const_wchar_t_param);

// CHECK: ParmVarDecl {{.+}} out_const_char_parm 'const char *__single __terminated_by(0)*__single'
void out_const_char_parm(const char **out_const_char_parm);

// CHECK: ParmVarDecl {{.+}} fptr 'const char *__single __terminated_by(0)(*__single)(const char *__single __terminated_by(0))'
void fptr_param(const char *(*fptr)(const char *p));

// CHECK: ParmVarDecl {{.+}} p1 'const char *__single __terminated_by(0)'
// CHECK: ParmVarDecl {{.+}} p2 'const char *__single __terminated_by(0)(*__single)(const char *__single __terminated_by(0), const char *__single __terminated_by(0))'
// CHECK: ParmVarDecl {{.+}} p3 'const char *__single __terminated_by(0)'
void multiple_parms(const char *p1, const char *(*p2)(const char *a, const char *b), const char *p3);

// CHECK: FunctionDecl {{.+}} const_char_ret 'const char *__single __terminated_by(0)(void)'
const char *const_char_ret(void);

// CHECK: FunctionDecl {{.+}} fptr_ret_proto 'const char *__single __terminated_by(0)(*__single(void))(const char *__single __terminated_by(0))'
const char *(*fptr_ret_proto(void))(const char *);

// CHECK: FunctionDecl {{.+}} fptr_ret_no_proto 'const char *__single __terminated_by(0)(*__single())(const char *__single __terminated_by(0))'
const char *(*fptr_ret_no_proto())(const char *);

void foo(void) {
  // CHECK: VarDecl {{.+}} local_fptr 'const char *__single __terminated_by(0)(*__single)(const char *__single __terminated_by(0))'
  const char *(*local_fptr)(const char *p);

  // CHECK: VarDecl {{.+}} ptr_const_char_local 'const char *__single __terminated_by(0)*__bidi_indexable'
  const char **ptr_const_char_local;

  // CHECK: VarDecl {{.+}} local_fptr_array 'const char *__single __terminated_by(0)(*__single[42])(const char *__single __terminated_by(0))'
  const char *(*local_fptr_array[42])(const char *p);

  // CHECK: VarDecl {{.+}} local_fptr_array_ptr 'const char *__single __terminated_by(0)(*__single(*__bidi_indexable)[42])(const char *__single __terminated_by(0))'
  const char *(*(*local_fptr_array_ptr)[42])(const char *p);
}

// CHECK: VarDecl {{.+}} const_char_global 'const char *__single __terminated_by(0)'
const char *const_char_global;

// CHECK: VarDecl {{.+}} global_fptr 'const char *__single __terminated_by(0)(*__single)(const char *__single __terminated_by(0))'
const char *(*global_fptr)(const char *p);

struct const_char_struct {
  // CHECK: FieldDecl {{.+}} const_char_field 'const char *__single __terminated_by(0)'
  const char *const_char_field;
};

typedef const char *my_func_t(const char *p);
typedef const char *(*my_func_ptr_t)(const char *p);

// CHECK: FunctionDecl {{.+}} typedef_func 'const char *__single __terminated_by(0)(const char *__single __terminated_by(0))'
my_func_t typedef_func;

// CHECK: VarDecl {{.+}} typedef_func_ptr 'const char *__single __terminated_by(0)(*__single)(const char *__single __terminated_by(0))'
my_func_ptr_t typedef_func_ptr;

// CHECK: ParmVarDecl {{.+}} quals_c 'const char *__single __terminated_by(0)const':'const char *__singleconst'
void quals_c(const char *const quals_c);

// CHECK: ParmVarDecl {{.+}} quals_cv 'const char *__single __terminated_by(0)const volatile':'const char *__singleconst volatile'
void quals_cv(const char *const volatile quals_cv);

/*
 * The cases where __null_terminated should NOT be added:
 */

void bar(void) {
  // CHECK: VarDecl {{.+}} const_char_local 'const char *__bidi_indexable'
  const char *const_char_local;

  // CHECK: VarDecl {{.+}} local_fptr 'const char *__single(*__single)(const char *__single __counted_by(8))'
  const char *__single (*local_fptr)(const char *__counted_by(8) p);
}

// CHECK: ParmVarDecl {{.+}} const_int_parm 'const int *__single'
void const_int_parm(const int *const_int_parm);

// CHECK: ParmVarDecl {{.+}} const_int8_t_param 'const int8_t *__single'
void const_int8_t_param(const int8_t *const_int8_t_param);

// CHECK: ParmVarDecl {{.+}} char_parm 'char *__single'
void char_parm(char *char_parm);

// CHECK: ParmVarDecl {{.+}} unsafe_parm 'const char *__unsafe_indexable'
void unsafe_parm(const char *__unsafe_indexable unsafe_parm);

// CHECK: ParmVarDecl {{.+}} single_parm 'const char *__single'
void single_parm(const char *__single single_parm);

// CHECK: ParmVarDecl {{.+}} bidi_parm 'const char *__bidi_indexable'
void bidi_parm(const char *__bidi_indexable bidi_parm);

// CHECK: ParmVarDecl {{.+}} counted_parm 'const char *__single __counted_by(8)'
void counted_parm(const char *__counted_by(8) counted_parm);

// CHECK: ParmVarDecl {{.+}} ended_parm 'const char *__single __ended_by(end)'
// CHECK: ParmVarDecl {{.+}} end 'const char *__single /* __started_by(ended_parm) */ ':'const char *__single'
void ended_parm_single(const char *__ended_by(end) ended_parm, const char *__single end);

// CHECK: ParmVarDecl {{.+}} ended_parm 'const char *__single __ended_by(end)'
// CHECK: ParmVarDecl {{.+}} end 'const char *__single /* __started_by(ended_parm) */ ':'const char *__single'
void ended_parm_unspec(const char *__ended_by(end) ended_parm, const char *end);

// CHECK: ParmVarDecl {{.+}} nt_parm 'const char *__single __terminated_by(0)'
void nt_parm(const char *__null_terminated nt_parm);

// CHECK: ParmVarDecl {{.+}} tb_parm 'const char *__single __terminated_by('X')'
void tb_parm(const char *__terminated_by('X') tb_parm);

// CHECK: ParmVarDecl {{.+}} out_single 'const char *__single*__single'
void out_single(const char *__single *out_single);

// CHECK: ParmVarDecl {{.+}} out_counted 'const char *__single __counted_by(8)*__single'
void out_counted(const char *__counted_by(8) * out_counted);

// CHECK: FunctionDecl {{.+}} system_func 'void (const char *)'
#include "auto-bound-const-char-pointer-param-system.h"

__ptrcheck_abi_assume_unsafe_indexable();

// CHECK: FunctionDecl {{.+}} unsafe_abi 'void (const char *__unsafe_indexable)'
void unsafe_abi(const char *p);

// Make sure we add __single to __terminated_by() even if the ABI is __unsafe_indexable.
// CHECK: FunctionDecl {{.+}} unsafe_abi_explicit 'void (const char *__single __terminated_by(0))'
void unsafe_abi_explicit(const char *__null_terminated p);
