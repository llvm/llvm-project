// Linkage types of global variables
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s -check-prefix=CIR --input-file %t.cir
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s -check-prefix=LLVM --input-file %t-cir.ll
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck %s -check-prefix=OGCG --input-file %t.ll

void a() {}

// CIR: cir.func dso_local @_Z1av()
// LLVM: define dso_local void @_Z1av()
// OGCG: define dso_local void @_Z1av()

extern void b();
// CIR: cir.func private @_Z1bv()
// LLVM: declare void @_Z1bv()
// OGCG: declare void @_Z1bv()

static void c() {}
// CIR: cir.func internal private dso_local @_ZL1cv()
// LLVM: define internal void @_ZL1cv()
// OGCG: define internal void @_ZL1cv()

inline void d() {}
// CIR: cir.func comdat linkonce_odr @_Z1dv()
// LLVM: define linkonce_odr void @_Z1dv()
// OGCG: define linkonce_odr void @_Z1dv(){{.*}} comdat

namespace {
  void e() {}
}

// CIR: cir.func internal private dso_local @_ZN12_GLOBAL__N_11eEv()
// LLVM: define internal void @_ZN12_GLOBAL__N_11eEv()
// OGCG: define internal void @_ZN12_GLOBAL__N_11eEv()

void f();
// CIR: cir.func private @_Z1fv()
// LLVM: declare void @_Z1fv()
// OGCG: declare void @_Z1fv()

// Force the functions to be emitted
void reference_funcs() {
    a();
    b();
    c();
    d();
    e();
    f();
}
