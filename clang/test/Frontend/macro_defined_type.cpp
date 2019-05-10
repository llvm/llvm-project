// RUN: %clang_cc1 -fsyntax-only -verify %s

#define NODEREF __attribute__((noderef))

void Func() {
  int NODEREF i; // expected-warning{{'noderef' can only be used on an array or pointer type}}
  int NODEREF *i_ptr;

  // There should be no difference whether a macro defined type is used or not.
  auto __attribute__((noderef)) *auto_i_ptr = i_ptr;
  auto __attribute__((noderef)) auto_i = i; // expected-warning{{'noderef' can only be used on an array or pointer type}}

  auto NODEREF *auto_i_ptr2 = i_ptr;
  auto NODEREF auto_i2 = i; // expected-warning{{'noderef' can only be used on an array or pointer type}}
}

// Added test for fix for P41835
#define _LIBCPP_FLOAT_ABI __attribute__((pcs("aapcs")))
struct A {
  _LIBCPP_FLOAT_ABI int operator()() throw(); // expected-warning{{'pcs' calling convention ignored for this target}}
};
