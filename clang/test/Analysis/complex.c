// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store=region -verify -Wno-unreachable-code -ffreestanding %s

#include <stdint.h>

void f1(int * p) {
  
  // This branch should be infeasible
  // because __imag__ p is 0.
  if (!p && __imag__ (intptr_t) p)
    *p = 1; // no-warning

  // If p != 0 then this branch is feasible; otherwise it is not.
  if (__real__ (intptr_t) p)
    *p = 1; // no-warning
    
  *p = 2; // expected-warning{{Dereference of null pointer}}
}
