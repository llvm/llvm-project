// RUN: %clang_analyze_cc1 -triple amdgcn-unknown-unknown \
// RUN: -analyze -analyzer-checker=core,alpha.unix.cstring \
// RUN: -analyze -analyzer-checker=debug.ExprInspection \
// RUN: -analyzer-config crosscheck-with-z3=true -verify %s \
// RUN: -Wno-incompatible-library-redeclaration
// REQUIRES: z3

void clang_analyzer_warnIfReached();

// From https://llvm.org/docs/AMDGPUUsage.html#address-spaces,
// select address space 3 (local), since the pointer size is
// different than Generic.
#define DEVICE __attribute__((address_space(3)))
_Static_assert(sizeof(int *) == 8, "");
_Static_assert(sizeof(DEVICE int *) == 4, "");
_Static_assert(sizeof(void *) == 8, "");
_Static_assert(sizeof(DEVICE void *) == 4, "");

// Copy from host to device memory. Note this is specialized
// since one of the parameters is assigned an address space such
// that the sizeof the the pointer is different than the other.
//
// Some downstream implementations may have specialized memcpy
// routines that copy from one address space to another. In cases
// like that, the address spaces are assumed to not overlap, so the
// cstring overlap check is not needed. When a static analysis report
// is generated in as case like this, SMTConv may attempt to create
// a refutation to Z3 with different bitwidth pointers which lead to
// a crash. This is not common in directly used upstream compiler builds,
// but can be seen in specialized downstrean implementations. This case
// covers those specific instances found and debugged.
//
// Since memcpy is a builtin, a specialized builtin instance named like
// 'memcpy_special' will hit in cstring, triggering this behavior. The
// best we can do for an upstream test is use the same memcpy function name.
DEVICE void *memcpy(DEVICE void *dst, const void *src, unsigned long len);

void top1(DEVICE void *dst, void *src, int len) {
  memcpy(dst, src, len);

  // Create a bugreport for triggering Z3 refutation.
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void top2(DEVICE int *dst, void *src, int len) {
  memcpy(dst, src, len);

  // Create a bugreport for triggering Z3 refutation.
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void top3(DEVICE int *dst, int *src, int len) {
  memcpy(dst, src, len);

  // Create a bugreport for triggering Z3 refutation.
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void top4() {
  memcpy((DEVICE void *)1, (const void *)1, 1);

  // Create a bugreport for triggering Z3 refutation.
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}
