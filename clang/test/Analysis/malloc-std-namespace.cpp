// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc -verify -analyzer-output=text %s

// This file tests that unix.Malloc can handle C++ code where e.g. malloc and
// free are declared within the namespace 'std' by the header <cstdlib>.

#include "Inputs/system-header-simulator-cxx.h"

void leak() {
  int *p = static_cast<int*>(std::malloc(sizeof(int))); // expected-note{{Memory is allocated}}
} // expected-warning{{Potential leak of memory pointed to by 'p'}}
  // expected-note@-1{{Potential leak of memory pointed to by 'p'}}

void no_leak() {
  int *p = static_cast<int*>(std::malloc(sizeof(int)));
  std::free(p); // no-warning
}

void invalid_free() {
  int i;
  int *p = &i;
  //expected-note@+2{{Argument to 'free()' is the address of the local variable 'i', which is not memory allocated by 'malloc()'}}
  //expected-warning@+1{{Argument to 'free()' is the address of the local variable 'i', which is not memory allocated by 'malloc()'}}
  std::free(p);
}
