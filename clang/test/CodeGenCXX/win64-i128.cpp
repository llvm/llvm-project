// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=GNU
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=MSVC

typedef int int128_t __attribute__((mode(TI)));

struct S {
  int128_t f();
};

int128_t S::f() { return 0; }

// GNU-LABEL: define dso_local void @_ZN1S1fEv(
// GNU-SAME: ptr dead_on_unwind noalias writable sret(i128) align 16 %agg.result,
// GNU-SAME: ptr noundef nonnull align 1 dereferenceable(1) %this)
// MSVC-LABEL: define dso_local void @"?f@S@@QEAA_LXZ"(
// MSVC-SAME: ptr noundef nonnull align 1 dereferenceable(1) %this,
// MSVC-SAME: ptr dead_on_unwind noalias writable sret(i128) align 16 %agg.result)
