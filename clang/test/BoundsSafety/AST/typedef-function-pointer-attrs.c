
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

typedef int *(*fp_proto_t)(int *p);
typedef int *(*fp_proto_va_t)(int *p, ...);
typedef int *(*fp_no_proto_t)();

// CHECK: VarDecl {{.+}} g_var_proto 'int *__single(*__single)(int *__single)'
fp_proto_t g_var_proto;

// CHECK: VarDecl {{.+}} g_var_proto_va 'int *__single(*__single)(int *__single, ...)'
fp_proto_va_t g_var_proto_va;

// CHECK: VarDecl {{.+}} g_var_no_proto 'int *__single(*__single)()'
fp_no_proto_t g_var_no_proto;

void foo(void) {
  // CHECK: VarDecl {{.+}} l_var_proto 'int *__single(*__single)(int *__single)'
  fp_proto_t l_var_proto;

  // CHECK: VarDecl {{.+}} l_var_proto_va 'int *__single(*__single)(int *__single, ...)'
  fp_proto_va_t l_var_proto_va;

  // CHECK: VarDecl {{.+}} l_var_no_proto 'int *__single(*__single)()'
  fp_no_proto_t l_var_no_proto;
}

// CHECK: ParmVarDecl {{.+}} param_proto 'int *__single(*__single)(int *__single)'
void f1(fp_proto_t param_proto);

// CHECK: ParmVarDecl {{.+}} param_proto_va 'int *__single(*__single)(int *__single, ...)'
void f2(fp_proto_va_t param_proto_va);

// CHECK: ParmVarDecl {{.+}} param_no_proto 'int *__single(*__single)()'
void f3(fp_no_proto_t param_no_proto);

// CHECK: FunctionDecl {{.+}} ret_proto 'int *__single(*__single(void))(int *__single)'
fp_proto_t ret_proto(void);

// CHECK: FunctionDecl {{.+}} ret_proto_va 'int *__single(*__single(void))(int *__single, ...)'
fp_proto_va_t ret_proto_va(void);

// CHECK: FunctionDecl {{.+}} ret_no_proto 'int *__single(*__single(void))()'
fp_no_proto_t ret_no_proto(void);

struct bar {
  // CHECK: FieldDecl {{.+}} field_proto 'int *__single(*__single)(int *__single)'
  fp_proto_t field_proto;

  // CHECK: FieldDecl {{.+}} field_proto_va 'int *__single(*__single)(int *__single, ...)'
  fp_proto_va_t field_proto_va;

  // CHECK: FieldDecl {{.+}} field_no_proto 'int *__single(*__single)()'
  fp_no_proto_t field_no_proto;
};

typedef fp_proto_t (*nested_t)(fp_proto_va_t p1, fp_no_proto_t p2);

// CHECK: VarDecl {{.+}} g_nested 'int *__single(*__single(*__single)(int *__single(*__single)(int *__single, ...), int *__single(*__single)()))(int *__single)'
nested_t g_nested;

typedef int *(f_proto_t)(int *p);
typedef int *(f_proto_va_t)(int *p, ...);
typedef int *(f_no_proto_t)();

// CHECK: FunctionDecl {{.+}} f_proto 'int *__single(int *__single)'
f_proto_t f_proto;

// CHECK: FunctionDecl {{.+}} f_proto_va 'int *__single(int *__single, ...)'
f_proto_va_t f_proto_va;

// CHECK: FunctionDecl {{.+}} f_no_proto 'int *__single()'
f_no_proto_t f_no_proto;
