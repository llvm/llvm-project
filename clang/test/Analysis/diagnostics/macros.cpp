// RUN: %clang_cc1 -analyze -std=c++11 -analyzer-checker=core,osx -analyzer-output=text -verify %s

#include "../Inputs/system-header-simulator.h"
#include "../Inputs/system-header-simulator-cxx.h"

void testIntMacro(unsigned int i) {
  if (i == UINT32_MAX) { // expected-note {{Assuming 'i' is equal to UINT32_MAX}}
                         // expected-note@-1 {{Taking true branch}}
    char *p = NULL; // expected-note {{'p' initialized to a null pointer value}}
    *p = 7;  // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
             // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  }
}

void testNULLMacro(int *p) {
  if (p == NULL) { // expected-note {{Assuming 'p' is equal to NULL}}
                   // expected-note@-1 {{Taking true branch}}
    *p = 7;  // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
             // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  }
}

void testnullptrMacro(int *p) {
  if (p == nullptr) { // expected-note {{Assuming pointer value is null}}
                      // expected-note@-1 {{Taking true branch}}
    *p = 7;  // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
             // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  }
}

// There are no path notes on the comparison to float types.
void testDoubleMacro(double d) {
  if (d == DBL_MAX) { // expected-note {{Taking true branch}}

    char *p = NULL; // expected-note {{'p' initialized to a null pointer value}}
    *p = 7;         // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
                    // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  }
}

void testboolMacro(bool b, int *p) {
  p = nullptr;      // expected-note {{Null pointer value stored to 'p'}}
  if (b == false) { // expected-note {{Assuming the condition is true}}
                    // expected-note@-1 {{Taking true branch}}
    *p = 7;         // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
                    // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  }
}
