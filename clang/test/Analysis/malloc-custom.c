// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc -Wno-incompatible-library-redeclaration -verify %s

// Various tests to make the analyzer is robust against custom
// redeclarations of memory routines.
//
// You wouldn't expect to see much of this in normal code, but, for example,
// CMake tests can generate these.

// expected-no-diagnostics

char alloca();
char malloc();
char realloc();
char kmalloc();
char valloc();
char calloc();

char free();
char kfree();

void testCustomArgumentlessAllocation() {
  alloca(); // no-crash
  malloc(); // no-crash
  realloc(); // no-crash
  kmalloc(); // no-crash
  valloc(); // no-crash
  calloc(); // no-crash

  free(); // no-crash
  kfree(); // no-crash
}

