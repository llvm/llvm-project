// Linkage types of global variables
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s -check-prefix=CIR --input-file %t.cir
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s -check-prefix=LLVM --input-file %t-cir.ll
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck %s -check-prefix=OGCG --input-file %t.ll

// Note: The differences between CHECK and OGCG represent incorrect IR lowering
//       with ClangIR enabled and will be fixed in a future patch.

int aaaa;
// CIR: cir.global external @aaaa
// LLVM: @aaaa = dso_local global i32 0
// OGCG: @aaaa = global i32 0

[[gnu::selectany]] int dddd;
// CIR: cir.global weak_odr @dddd
// LLVM: @dddd = weak_odr dso_local global i32 0
// OGCG: @dddd = weak_odr global i32 0, comdat

static int bbbb;
// CIR: cir.global internal @_ZL4bbbb
// LLVM: @_ZL4bbbb = internal global i32 0
// OGCG: @_ZL4bbbb = internal global i32 0

inline int cccc;
// CIR: cir.global linkonce_odr @cccc
// LLVM: @cccc = linkonce_odr dso_local global i32 0
// OGCG: @cccc = linkonce_odr global i32 0

// Force the global variables to be emitted
void reference_vars() {
    bbbb;
    cccc;
}
