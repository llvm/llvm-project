// RUN: %clang_cc1 -triple spir64-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL3.0 %s

// Check that returning a pointer to a local address space variable does not
// trigger -Wreturn-stack-address.

// expected-no-diagnostics

#pragma OPENCL EXTENSION __cl_clang_function_scope_local_variables : enable

local int* get_group_scratch() {
  local int data[64];
  return data;
}
