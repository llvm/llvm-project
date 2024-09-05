// RUN: %clang_cc1 -I%S/Inputs -fsyntax-only -verify -triple x86_64-linux-gnu %s

#include <macro_defined_type.h>

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

// The diagnostic message is hard-coded as 'noderef' so using a system macro doesn't change the behavior
void Func_system_macro() {
  int _SYS_NODEREF i; // expected-warning{{'noderef' can only be used on an array or pointer type}}
  int _SYS_NODEREF *i_ptr;

  auto _SYS_NODEREF *auto_i_ptr2 = i_ptr;
  auto _SYS_NODEREF auto_i2 = i; // expected-warning{{'noderef' can only be used on an array or pointer type}}
}


// Added test for fix for P41835
#define _LIBCPP_FLOAT_ABI __attribute__((pcs("aapcs")))
struct A {
  _LIBCPP_FLOAT_ABI int operator()() throw(); // expected-warning{{'pcs' calling convention is not supported for this target}}
};

struct A_system_macro {
  _SYS_LIBCPP_FLOAT_ABI int operator()() throw(); // expected-warning{{'_SYS_LIBCPP_FLOAT_ABI' calling convention is not supported for this target}}
};

// Added test for fix for PR43315
#define a __attribute__((__cdecl__, __regparm__(0)))
int(a b)();
