// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// This would previously emit a "definition with same mangled name as another
// definition" error: https://github.com/llvm/clangir/issues/991.
namespace N {
struct S {
  // CHECK: cir.func linkonce_odr @_ZN1N1S3fooEv({{.*}} {
  void foo() {}
};

// CHECK: cir.func @_ZN1N1fEv() {{.*}} {
// CHECK:   cir.call @_ZN1N1S3fooEv(
void f() { S().foo(); }
} // namespace N
