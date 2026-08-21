// A __ptrauth-qualified pointer is loaded through its own path in
// CGPointerAuth.cpp, which needs a triple with pointer authentication, hence a
// file of its own. See ubsan-array-bounds-baseline.c for the rest.
//
// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64e-apple-macosx11.0.0 -fptrauth-calls \
// RUN:     -fptrauth-intrinsics -fptrauth-returns -emit-llvm \
// RUN:     -fsanitize=array-bounds -Wno-array-bounds %s -o - | FileCheck %s

#define AQ __ptrauth(2, 1, 42)

int *AQ pa[4];

// CHECK-LABEL: define {{.*}}@p_load(
// CHECK: icmp ult i64 {{.*}}, 4
int *p_load(int i) { return pa[i]; }

// CHECK-LABEL: define {{.*}}@p_load_paren(
// CHECK: icmp ult i64 {{.*}}, 4
int *p_load_paren(int i) { return (pa[i]); }

// CHECK-LABEL: define {{.*}}@p_load_deref_addr(
// CHECK: icmp ule i64 {{.*}}, 4
int *p_load_deref_addr(int i) { return *&pa[i]; }
