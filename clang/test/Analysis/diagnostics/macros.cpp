// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,osx -analyzer-output=text -verify %s

#include "../Inputs/system-header-simulator.h"
#include "../Inputs/system-header-simulator-cxx.h"

void testUnsignedIntMacro(unsigned int i) {
  if (i == UINT32_MAX) { // expected-note {{Assuming 'i' is equal to UINT32_MAX}}
                         // expected-note@-1 {{Taking true branch}}
    char *p = NULL; // expected-note {{'p' initialized to a null pointer value}}
    *p = 7;  // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
             // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  }
}


// FIXME: 'i' can never be equal to UINT32_MAX - it doesn't even fit into its
// type ('int'). This should say "Assuming 'i' is equal to -1".
void testIntMacro(int i) {
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
  if (d == DBL_MAX) { // expected-note {{Assuming 'd' is equal to DBL_MAX}}
                      // expected-note@-1 {{Taking true branch}}

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

#define nested_null_split(x) if ((x) != UINT32_MAX) {}

void testNestedNullSplitMacro(int i, int *p) {
  nested_null_split(i); // expected-note {{Assuming 'i' is equal to -1}}
                        // expected-note@-1 {{Taking false branch}}
  if (!p) // expected-note {{Assuming 'p' is null}}
          // expected-note@-1 {{Taking true branch}}
    *p = 1; // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
            // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
}
