// Refer issue 55019 for more details.
// A supplemental test case of pr22954.c for other functions modeled in
// the CStringChecker.

// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix \
// RUN:   -analyzer-checker=debug.ExprInspection

#include "Inputs/system-header-simulator.h"
#include "Inputs/system-header-simulator-cxx.h"

void *malloc(size_t);
void free(void *);

struct mystruct {
  void *ptr;
  char arr[4];
};

void clang_analyzer_dump(const void *);

// CStringChecker::memsetAux
void fmemset() {
  mystruct x;
  x.ptr = malloc(1);
  clang_analyzer_dump(x.ptr); // expected-warning {{HeapSymRegion}}
  memset(x.arr, 0, sizeof(x.arr));
  clang_analyzer_dump(x.ptr); // expected-warning {{HeapSymRegion}}
  free(x.ptr);                // no-leak-warning
}

// CStringChecker::evalCopyCommon
void fmemcpy() {
  mystruct x;
  x.ptr = malloc(1);
  clang_analyzer_dump(x.ptr); // expected-warning {{HeapSymRegion}}
  memcpy(x.arr, "hi", 2);
  clang_analyzer_dump(x.ptr); // expected-warning {{HeapSymRegion}}
  free(x.ptr);                // no-leak-warning
}

// CStringChecker::evalStrcpyCommon
void fstrcpy() {
  mystruct x;
  x.ptr = malloc(1);
  clang_analyzer_dump(x.ptr); // expected-warning {{HeapSymRegion}}
  strcpy(x.arr, "hi");
  clang_analyzer_dump(x.ptr); // expected-warning {{HeapSymRegion}}
  free(x.ptr);                // no-leak-warning
}

void fstrncpy() {
  mystruct x;
  x.ptr = malloc(1);
  clang_analyzer_dump(x.ptr); // expected-warning {{HeapSymRegion}}
  strncpy(x.arr, "hi", sizeof(x.arr));
  clang_analyzer_dump(x.ptr); // expected-warning {{HeapSymRegion}}
  free(x.ptr);                // no-leak-warning
}

// CStringChecker::evalStrsep
void fstrsep() {
  mystruct x;
  x.ptr = malloc(1);
  clang_analyzer_dump(x.ptr); // expected-warning {{HeapSymRegion}}
  char *p = x.arr;
  (void)strsep(&p, "x");
  clang_analyzer_dump(x.ptr); // expected-warning {{HeapSymRegion}}
  free(x.ptr);                // no-leak-warning
}

// CStringChecker::evalStdCopyCommon
void fstdcopy() {
  mystruct x;
  x.ptr = new char;
  clang_analyzer_dump(x.ptr); // expected-warning {{HeapSymRegion}}

  const char *p = "x";
  std::copy(p, p + 1, x.arr);

  // FIXME: As we currently cannot know whether the copy overflows, the checker
  // invalidates the entire `x` object. When the copy size through iterators
  // can be correctly modeled, we can then update the verify direction from
  // SymRegion to HeapSymRegion as this std::copy call never overflows and
  // hence the pointer `x.ptr` shall not be invalidated.
  clang_analyzer_dump(x.ptr);       // expected-warning {{SymRegion}}
  delete static_cast<char*>(x.ptr); // no-leak-warning
}
