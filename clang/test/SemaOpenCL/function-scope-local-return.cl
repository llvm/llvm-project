// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL3.0

// Check returning a pointer to a local address space variable does not warn
// if __cl_clang_function_scope_local_variables extension is enabled.

// expected-no-diagnostics

#pragma OPENCL EXTENSION __cl_clang_function_scope_local_variables : enable

local int* get_group_scratch() {
  local int data[64];
  return data;
}
