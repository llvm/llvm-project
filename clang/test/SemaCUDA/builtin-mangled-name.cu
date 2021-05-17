// RUN: %clang_cc1 -triple x86_64-unknown-gnu-linux -aux-triple amdgcn-amd-amdhsa \
// RUN:   -verify -fsyntax-only -x hip %s

#include "Inputs/cuda.h"

__global__ void kern1();
int y;

void fun1() {
  int x;
  const char *p;
  p = __builtin_get_device_side_mangled_name();
  // expected-error@-1 {{invalid argument: symbol must be a device-side function or global variable}}
  p = __builtin_get_device_side_mangled_name(kern1, kern1);
  // expected-error@-1 {{invalid argument: symbol must be a device-side function or global variable}}
  p = __builtin_get_device_side_mangled_name(1);
  // expected-error@-1 {{invalid argument: symbol must be a device-side function or global variable}}
  p = __builtin_get_device_side_mangled_name(x);
  // expected-error@-1 {{invalid argument: symbol must be a device-side function or global variable}}
  p = __builtin_get_device_side_mangled_name(fun1);
  // expected-error@-1 {{invalid argument: symbol must be a device-side function or global variable}}
  p = __builtin_get_device_side_mangled_name(y);
  // expected-error@-1 {{invalid argument: symbol must be a device-side function or global variable}}
}
