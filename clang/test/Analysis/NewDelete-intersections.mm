// RUN: %clang_analyze_cc1 -std=c++11 -fblocks %s \
// RUN:  -Wno-alloc-size \
// RUN:  -verify=newdelete \
// RUN:  -analyzer-checker=core \
// RUN:  -analyzer-checker=cplusplus.NewDelete

// leak-no-diagnostics

// RUN: %clang_analyze_cc1 -std=c++11 -DLEAKS -fblocks %s \
// RUN:   -Wno-alloc-size \
// RUN:   -verify=leak \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDeleteLeaks

// RUN: %clang_analyze_cc1 -std=c++11 -DLEAKS -fblocks %s \
// RUN:   -Wno-alloc-size \
// RUN:   -verify=mismatch \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.MismatchedDeallocator

#include "Inputs/system-header-simulator-cxx.h"
#include "Inputs/system-header-simulator-objc.h"

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);
extern "C" void *alloca(size_t);
extern "C" void free(void *);

void testMallocFreeNoWarn() {
  int *p1 = (int *)malloc(sizeof(int));
  free(++p1); // no warn

  int *p2 = (int *)malloc(sizeof(int));
  free(p2);
  free(p2); // no warn

  int *p3 = (int *)malloc(sizeof(int)); // no warn

  int *p4 = (int *)malloc(sizeof(int));
  free(p4);
  int j = *p4; // no warn

  int *p5 = (int *)alloca(sizeof(int));
  free(p5); // no warn
}

void testDeleteMalloced() {
  int *p1 = (int *)malloc(sizeof(int));
  delete p1;
  // mismatch-warning@-1{{Memory allocated by 'malloc()' should be deallocated by 'free()', not 'delete'}}

  int *p2 = (int *)__builtin_alloca(sizeof(int));
  delete p2; // no warn
}

void testUseZeroAllocatedMalloced() {
  int *p1 = (int *)malloc(0);
  *p1 = 1; // no warn
}

//----- Test free standard new
void testFreeOpNew() {
  void *p = operator new(0);
  free(p);
  // mismatch-warning@-1{{Memory allocated by 'operator new' should be deallocated by 'delete', not 'free()'}}
}

void testFreeNewExpr() {
  int *p = new int;
  free(p);
  // mismatch-warning@-1{{Memory allocated by 'new' should be deallocated by 'delete', not 'free()'}}
  free(p);
}

void testObjcFreeNewed() {
  int *p = new int;
  NSData *nsdata = [NSData dataWithBytesNoCopy:p length:sizeof(int) freeWhenDone:1];
  // mismatch-warning@-1{{+dataWithBytesNoCopy:length:freeWhenDone: cannot take ownership of memory allocated by 'new'}}
}

void testFreeAfterDelete() {
  int *p = new int;
  delete p;
  free(p); // newdelete-warning{{Use of memory after it is released}}
}

void testStandardPlacementNewAfterDelete() {
  int *p = new int;
  delete p;
  p = new (p) int; // newdelete-warning{{Use of memory after it is released}}
}
