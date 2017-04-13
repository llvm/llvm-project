// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete,cplusplus.NewDeleteLeaks -std=c++11 -verify %s

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

//-------------------------------------------------------------------
// Check that unix.Malloc + cplusplus.NewDelete does not enable
// warnings produced by unix.MismatchedDeallocator.
//-------------------------------------------------------------------
void testMismatchedDeallocator() {
  int *p = (int *)malloc(sizeof(int));
  delete p;
} // expected-warning{{Potential leak of memory pointed to by 'p'}}
