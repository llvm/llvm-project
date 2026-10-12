// REQUIRES: asserts
// RUN: not --crash %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown \
// RUN:   -emit-llvm -o - %s -DNONCONST 2>&1 | FileCheck %s --check-prefix=CHECK-NONCONST
// RUN: not --crash %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown \
// RUN:   -emit-llvm -o - %s -DCONST 2>&1 | FileCheck %s --check-prefix=CHECK-CONST

// static_cast<T&&> of a bit-field lvalue should materialize a temporary.
// FIXME: Sema binds the reference directly to the bit-field, so CodeGen
// receives a bit-field LValue and asserts in EmitReferenceBindingToExpr.
// Each case is guarded by its own macro because the crash aborts the whole
// compilation.

struct A {
  int a : 4;
};

// CHECK-LABEL: define {{.*}} @_Z4test1A(
void test(A obj) {
  #ifdef NONCONST
  // CHECK-NONCONST: {{Assertion.*LV.isSimple\(\)}}
  int &&ri = static_cast<int&&>(obj.a);
  #endif

#ifdef CONST
  // CHECK-CONST: {{Assertion.*LV.isSimple\(\)}}
  const int &&cri = static_cast<const int &&>(obj.a);
  #endif
}
