
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK: VarDecl {{.+}} global 'int *__single'
int *global;

// CHECK: VarDecl {{.+}} static_global 'int *__single' static
static int *static_global;

// CHECK: VarDecl {{.+}} extern_global 'int *__single' extern
extern int *extern_global;

// CHECK: VarDecl {{.+}} private_extern_global 'int *__single' __private_extern__
__private_extern__ int *private_extern_global;

// CHECK: VarDecl {{.+}} thread_global 'int *__single' tls
__thread int *thread_global;

// CHECK: VarDecl {{.+}} static_thread_global 'int *__single' static tls
static __thread int *static_thread_global;

// CHECK: VarDecl {{.+}} extern_thread_global 'int *__single' extern tls
extern __thread int *extern_thread_global;

// CHECK: VarDecl {{.+}} private_extern_thread_global 'int *__single' __private_extern__ tls
__private_extern__ __thread int *private_extern_thread_global;

void foo(void) {
  // CHECK: VarDecl {{.+}} local 'int *__bidi_indexable'
  int *local;

  // CHECK: VarDecl {{.+}} static_local 'int *__single' static
  static int *static_local;

  // CHECK: VarDecl {{.+}} extern_local 'int *__single' extern
  extern int *extern_local;

  // CHECK: VarDecl {{.+}} private_extern_local 'int *__single' __private_extern__
  __private_extern__ int *private_extern_local;

  // CHECK: VarDecl {{.+}} static_thread_local 'int *__single' static tls
  static __thread int *static_thread_local;

  // CHECK: VarDecl {{.+}} extern_thread_local 'int *__single' extern tls
  extern __thread int *extern_thread_local;

  // CHECK: VarDecl {{.+}} private_extern_thread_local 'int *__single' __private_extern__ tls
  __private_extern__ __thread int *private_extern_thread_local;
}
