// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=c++

// Taken from opencl-c.h
#define CLK_NULL_EVENT (__builtin_astype(((void*)(__SIZE_MAX__)), clk_event_t))

global clk_event_t ce; // expected-error {{the '__global clk_event_t' type cannot be used to declare a program scope variable}}

int clk_event_tests() {
  event_t e;
  clk_event_t ce1;
  clk_event_t ce2;

  if (e == ce1) { // expected-error {{invalid operands to binary expression ('event_t' and 'clk_event_t')}}
    return 9;
  }

  if (ce1 != ce2) {
    return 1;
  }
  else if (ce1 == CLK_NULL_EVENT || ce2 != CLK_NULL_EVENT) {
    return 0;
  }

  return 2;
}
