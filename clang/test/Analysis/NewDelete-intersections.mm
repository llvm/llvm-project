// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDelete -std=c++11 -fblocks -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDelete,cplusplus.NewDeleteLeaks -std=c++11 -DLEAKS -fblocks -verify %s
#include "Inputs/system-header-simulator-cxx.h"
#include "Inputs/system-header-simulator-objc.h"

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);
extern "C" void *alloca(size_t);
extern "C" void free(void *);

//----------------------------------------------------------------------------
// Check for intersections with unix.Malloc and unix.MallocWithAnnotations 
// checkers bounded with cplusplus.NewDelete.
//----------------------------------------------------------------------------

//----- malloc()/free() are subjects of unix.Malloc and unix.MallocWithAnnotations
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

  int *p5 = (int *)alloca(sizeof(int));
  free(p5); // no warn
}

void testDeleteMalloced() {
  int *p1 = (int *)malloc(sizeof(int));
  delete p1; // no warn

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
}
#ifdef LEAKS
// expected-warning@-2 {{Potential leak of memory pointed to by 'p'}}
#endif

void testFreeNewExpr() {
  int *p = new int;
  free(p);
}
#ifdef LEAKS
// expected-warning@-2 {{Potential leak of memory pointed to by 'p'}}
#endif

void testObjcFreeNewed() {
  int *p = new int;
  NSData *nsdata = [NSData dataWithBytesNoCopy:p length:sizeof(int) freeWhenDone:1];
#ifdef LEAKS
  // expected-warning@-2 {{Potential leak of memory pointed to by 'p'}}
#endif
}

void testFreeAfterDelete() {
  int *p = new int;  
  delete p;
  free(p); // expected-warning{{Use of memory after it is freed}}
}

void testStandardPlacementNewAfterDelete() {
  int *p = new int;  
  delete p;
  p = new(p) int; // expected-warning{{Use of memory after it is freed}}
}
