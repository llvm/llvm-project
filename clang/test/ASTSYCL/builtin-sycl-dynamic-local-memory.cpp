// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -ast-dump %s | FileCheck %s

// Check that a call to __builtin_sycl_dynamic_local_memory has type
// 'unsigned char [[clang::sycl_local]] *'.

auto f() {
  return __builtin_sycl_dynamic_local_memory();
}
// CHECK:      FunctionDecl {{.*}} f '{{\[\[}}clang::sycl_local{{\]\]}} unsigned char *()'
// CHECK:      ReturnStmt
// CHECK-NEXT: CallExpr {{.*}} '{{\[\[}}clang::sycl_local{{\]\]}} unsigned char *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} <BuiltinFnToFnPtr>
// CHECK-NEXT: DeclRefExpr {{.*}} Function {{.*}} '__builtin_sycl_dynamic_local_memory'
