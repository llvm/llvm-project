// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/232250.
// With a 32-bit size_t, 'new S[0xFFFFFFFF]' has no multiplication overflow
// (sizeof(S) == 1), but adding the array cookie overflows. The allocation
// size becomes all-ones, which can coincide with the size without the cookie.
// This used to hit an assertion in EmitCXXNewExpr.

struct S {
  ~S();
};

// CHECK-LABEL: define {{.*}} @_Z1fv(
// CHECK: call {{.*}} @_Znaj(i32 {{.*}}-1)
void f() { new S[0xFFFFFFFFu]; }
