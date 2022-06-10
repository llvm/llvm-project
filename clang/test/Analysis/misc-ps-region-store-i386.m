// RUN: %clang_analyze_cc1 -triple i386-apple-darwin9 -analyzer-checker=core,alpha.core -analyzer-store=region -verify -fblocks %s
// expected-no-diagnostics

// Here is a case where a pointer is treated as integer, invalidated as an
// integer, and then used again as a pointer.   This test just makes sure
// we don't crash.
typedef unsigned uintptr_t;
void test_pointer_invalidated_as_int_aux(uintptr_t* ptr);
void test_pointer_invalidated_as_int(void) {
  void *x;
  test_pointer_invalidated_as_int_aux((uintptr_t*) &x);
  // Here we have a pointer to integer cast.
  uintptr_t y = (uintptr_t) x;
}

