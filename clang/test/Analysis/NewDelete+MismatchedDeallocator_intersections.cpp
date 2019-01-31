// RUN: %clang_analyze_cc1 -std=c++11 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-checker=unix.MismatchedDeallocator
//
// RUN: %clang_analyze_cc1 -std=c++11 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-checker=cplusplus.NewDeleteLeaks \
// RUN:   -analyzer-checker=unix.MismatchedDeallocator

// expected-no-diagnostics

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

//------------------------------------------------------------------
// Check that alpha.cplusplus.NewDelete + unix.MismatchedDeallocator 
// does not enable warnings produced by the unix.Malloc checker.
//------------------------------------------------------------------
void testMallocFreeNoWarn() {
  int i;
  free(&i); // no warn

  int *p1 = (int *)malloc(sizeof(int));
  free(++p1); // no warn

  int *p2 = (int *)malloc(sizeof(int));
  free(p2);
  free(p2); // no warn

  int *p3 = (int *)malloc(sizeof(int)); // no warn

  int *p4 = (int *)malloc(sizeof(int));
  free(p4);
  int j = *p4; // no warn
}
