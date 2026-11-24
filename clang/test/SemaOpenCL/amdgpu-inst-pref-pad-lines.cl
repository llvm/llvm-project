// RUN: %clang_cc1 -triple amdgpu-- -verify -fsyntax-only %s

typedef __attribute__((amdgpu_inst_pref_pad_lines(2))) struct struct_pad { // expected-error {{'amdgpu_inst_pref_pad_lines' attribute only applies to kernel functions}}
  int x;
} struct_pad;

__attribute__((amdgpu_inst_pref_pad_lines(2))) void not_a_kernel() {} // expected-error {{'amdgpu_inst_pref_pad_lines' attribute only applies to kernel functions}}

__attribute__((amdgpu_inst_pref_pad_lines("ABC"))) kernel void not_an_int() {} // expected-error {{'amdgpu_inst_pref_pad_lines' attribute requires an integer constant}}

__attribute__((amdgpu_inst_pref_pad_lines(4294967296))) kernel void too_wide() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}

__attribute__((amdgpu_inst_pref_pad_lines(0))) kernel void zero() {}
__attribute__((amdgpu_inst_pref_pad_lines(2))) kernel void two() {}
