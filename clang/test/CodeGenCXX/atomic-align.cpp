// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -o - -triple=x86_64-linux-gnu | FileCheck %s

struct AM {
  int f1, f2;
};
alignas(8) AM m;
AM load1() {
  AM am;
  // m is declared to align to 8bytes, so generate load atomic instead
  // of libcall.
  // CHECK-LABEL: @_Z5load1v
  // CHECK: load atomic {{.*}} monotonic
  __atomic_load(&m, &am, 0);
  return am;
}

struct BM {
  int f1;
  alignas(8) AM f2;
};
BM bm;
AM load2() {
  AM am;
  // BM::f2 is declared to align to 8bytes, so generate load atomic instead
  // of libcall.
  // CHECK-LABEL: @_Z5load2v
  // CHECK: load atomic {{.*}} monotonic
  __atomic_load(&bm.f2, &am, 0);
  return am;
}
