// REQUIRES: asserts
// RUN: not --crash %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown \
// RUN:   -emit-llvm -O -disable-llvm-passes -o - %s -DNONCONST 2>&1 | FileCheck %s --check-prefix=CHECK-NONCONST
// RUN: not --crash %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown \
// RUN:   -emit-llvm -O -disable-llvm-passes -o - %s -DCONST 2>&1 | FileCheck %s --check-prefix=CHECK-CONST

// static_cast<T&&> of a bit-field lvalue should materialize a temporary.
// FIXME: Sema binds the reference directly to the bit-field, so CodeGen
// receives a bit-field LValue and asserts in EmitReferenceBindingToExpr.
// Each case is guarded by its own macro because the crash aborts the whole
// compilation.
//
// The tests are compiled with -O -disable-llvm-passes so that the lifetime
// markers for the temporary, as determined by the frontend, are emitted and
// can be observed once the crash is fixed. When fixed, these checks should
// verify that llvm.lifetime.start and llvm.lifetime.end are emitted for the
// materialized temporary.

struct A {
  int a : 4;
};

void test(A obj) {
#ifdef NONCONST
  // CHECK-NONCONST: {{Assertion.*LV.isSimple\(\)}}
  int &&ri = static_cast<int &&>(obj.a);
#endif

#ifdef CONST
  // CHECK-CONST: {{Assertion.*LV.isSimple\(\)}}
  const int &&cri = static_cast<const int &&>(obj.a);
#endif
}
