// RUN: %clang_analyze_cc1 -triple amdgcn-unknown-unknown \
// RUN: -analyze -analyzer-checker=core -DAMDGCN_TRIPLE \
// RUN: -analyze -analyzer-checker=debug.ExprInspection \
// RUN: -Wno-implicit-int -Wno-int-conversion -verify %s
//
// RUN: %clang_analyze_cc1 -triple amdgcn-unknown-unknown \
// RUN: -analyze -analyzer-checker=core -DDEFAULT_TRIPLE \
// RUN: -analyze -analyzer-checker=debug.ExprInspection \
// RUN: -Wno-implicit-int -Wno-int-conversion -verify %s

// From https://llvm.org/docs/AMDGPUUsage.html#address-spaces,
// select address space 3 (local), since the pointer size is
// different than Generic.

// expected-no-diagnostics

#define DEVICE __attribute__((address_space(3)))

#if defined(AMDGCN_TRIPLE)
// this crashes
int fn1() {
  int val = 0;
  DEVICE int *dptr = val;
  return dptr == (void *)0;
}

// does not crash
int fn2() {
  int val = 0;
  DEVICE int *dptr = val;
  return dptr == (DEVICE void *)0;
}

// this crashes
int fn3() {
  int val = 0;
  int *dptr = val;
  return dptr == (DEVICE void *)0;
}
#endif

// does not crash
int fn4() {
  int val = 0;
  int *dptr = val;
  return dptr == (void *)0;
}
