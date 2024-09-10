// RUN: %clang_cc1 -I%S/Inputs -fsyntax-only -triple x86_64-linux-gnu %s 2>&1 | FileCheck %s

#include <macro_defined_type.h>

// The diagnostic message is hard-coded as 'noderef' so using a system macro doesn't change the behavior
void Func_system_macro() {
  int _SYS_NODEREF i; // expected-warning{{'noderef' can only be used on an array or pointer type}}
  // CHECK: :[[@LINE-1]]:{{.*}} warning: 'noderef' can only be used on an array or pointer type

  int _SYS_NODEREF *i_ptr;

  auto _SYS_NODEREF *auto_i_ptr2 = i_ptr;
  auto _SYS_NODEREF auto_i2 = i; // expected-warning{{'noderef' can only be used on an array or pointer type}}
}

struct A_system_macro {
  _SYS_LIBCPP_FLOAT_ABI int operator()() throw();
  // CHECK: :[[@LINE-1]]:{{.*}} warning: '_SYS_LIBCPP_FLOAT_ABI' calling convention is not supported for this target
  // CHECK: {{.*}}macro_defined_type.h:{{.*}}:{{.*}}: note: expanded from macro '_SYS_LIBCPP_FLOAT_ABI'
  // CHECK:  4 | #define _SYS_LIBCPP_FLOAT_ABI __attribute__((pcs("aapcs")))
  // CHECK:    |                                              ^
};