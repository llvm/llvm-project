// RUN: %clang_cc1 -triple amdgcn-- -verify -fsyntax-only %s

typedef __attribute__((amdgpu_flat_work_group_size(32, 64))) struct struct_flat_work_group_size_32_64 { // expected-error {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
  int x;
  float y;
} struct_flat_work_group_size_32_64;
typedef __attribute__((amdgpu_waves_per_eu(2))) struct struct_waves_per_eu_2 { // expected-error {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
  int x;
  float y;
} struct_waves_per_eu_2;
typedef __attribute__((amdgpu_waves_per_eu(2, 4))) struct struct_waves_per_eu_2_4 { // expected-error {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
  int x;
  float y;
} struct_waves_per_eu_2_4;
typedef __attribute__((amdgpu_num_sgpr(32))) struct struct_num_sgpr_32 { // expected-error {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
  int x;
  float y;
} struct_num_sgpr_32;
typedef __attribute__((amdgpu_num_vgpr(64))) struct struct_num_vgpr_64 { // expected-error {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
  int x;
  float y;
} struct_num_vgpr_64;

__attribute__((amdgpu_flat_work_group_size(32, 64))) void func_flat_work_group_size_32_64() {} // expected-error {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
__attribute__((amdgpu_waves_per_eu(2))) void func_waves_per_eu_2() {} // expected-error {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
__attribute__((amdgpu_waves_per_eu(2, 4))) void func_waves_per_eu_2_4() {} // expected-error {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
__attribute__((amdgpu_num_sgpr(32))) void func_num_sgpr_32() {} // expected-error {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_num_vgpr(64))) void func_num_vgpr_64() {} // expected-error {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}

__attribute__((amdgpu_flat_work_group_size("ABC", "ABC"))) kernel void kernel_flat_work_group_size_ABC_ABC() {} // expected-error {{'amdgpu_flat_work_group_size' attribute requires an integer constant}}
__attribute__((amdgpu_flat_work_group_size(32, "ABC"))) kernel void kernel_flat_work_group_size_32_ABC() {} // expected-error {{'amdgpu_flat_work_group_size' attribute requires an integer constant}}
__attribute__((amdgpu_flat_work_group_size("ABC", 64))) kernel void kernel_flat_work_group_size_ABC_64() {} // expected-error {{'amdgpu_flat_work_group_size' attribute requires an integer constant}}
__attribute__((amdgpu_waves_per_eu("ABC"))) kernel void kernel_waves_per_eu_ABC() {} // expected-error {{'amdgpu_waves_per_eu' attribute requires an integer constant}}
__attribute__((amdgpu_waves_per_eu(2, "ABC"))) kernel void kernel_waves_per_eu_2_ABC() {} // expected-error {{'amdgpu_waves_per_eu' attribute requires an integer constant}}
__attribute__((amdgpu_waves_per_eu("ABC", 4))) kernel void kernel_waves_per_eu_ABC_4() {} // expected-error {{'amdgpu_waves_per_eu' attribute requires an integer constant}}
__attribute__((amdgpu_num_sgpr("ABC"))) kernel void kernel_num_sgpr_ABC() {} // expected-error {{'amdgpu_num_sgpr' attribute requires an integer constant}}
__attribute__((amdgpu_num_vgpr("ABC"))) kernel void kernel_num_vgpr_ABC() {} // expected-error {{'amdgpu_num_vgpr' attribute requires an integer constant}}

__attribute__((amdgpu_flat_work_group_size(4294967296, 4294967296))) kernel void kernel_flat_work_group_size_L_L() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__attribute__((amdgpu_flat_work_group_size(32, 4294967296))) kernel void kernel_flat_work_group_size_32_L() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__attribute__((amdgpu_flat_work_group_size(4294967296, 64))) kernel void kernel_flat_work_group_size_L_64() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__attribute__((amdgpu_waves_per_eu(4294967296))) kernel void kernel_waves_per_eu_L() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__attribute__((amdgpu_waves_per_eu(2, 4294967296))) kernel void kernel_waves_per_eu_2_L() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__attribute__((amdgpu_waves_per_eu(4294967296, 4))) kernel void kernel_waves_per_eu_L_4() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__attribute__((amdgpu_num_sgpr(4294967296))) kernel void kernel_num_sgpr_L() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__attribute__((amdgpu_num_vgpr(4294967296))) kernel void kernel_num_vgpr_L() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}

__attribute__((amdgpu_flat_work_group_size(0, 64))) kernel void kernel_flat_work_group_size_0_64() {} // expected-error {{'amdgpu_flat_work_group_size' attribute argument is invalid: max must be 0 since min is 0}}
__attribute__((amdgpu_waves_per_eu(0, 4))) kernel void kernel_waves_per_eu_0_4() {} // expected-error {{'amdgpu_waves_per_eu' attribute argument is invalid: max must be 0 since min is 0}}

__attribute__((amdgpu_flat_work_group_size(64, 32))) kernel void kernel_flat_work_group_size_64_32() {} // expected-error {{'amdgpu_flat_work_group_size' attribute argument is invalid: min must not be greater than max}}
__attribute__((amdgpu_waves_per_eu(4, 2))) kernel void kernel_waves_per_eu_4_2() {} // expected-error {{'amdgpu_waves_per_eu' attribute argument is invalid: min must not be greater than max}}

__attribute__((amdgpu_waves_per_eu(2, 4, 8))) kernel void kernel_waves_per_eu_2_4_8() {} // expected-error {{'amdgpu_waves_per_eu' attribute takes no more than 2 arguments}}

__attribute__((amdgpu_flat_work_group_size(0, 0))) kernel void kernel_flat_work_group_size_0_0() {}
__attribute__((amdgpu_waves_per_eu(0))) kernel void kernel_waves_per_eu_0() {}
__attribute__((amdgpu_waves_per_eu(0, 0))) kernel void kernel_waves_per_eu_0_0() {}
__attribute__((amdgpu_num_sgpr(0))) kernel void kernel_num_sgpr_0() {}
__attribute__((amdgpu_num_vgpr(0))) kernel void kernel_num_vgpr_0() {}

kernel __attribute__((amdgpu_flat_work_group_size(32, 64))) void kernel_flat_work_group_size_32_64() {}
kernel __attribute__((amdgpu_waves_per_eu(2))) void kernel_waves_per_eu_2() {}
kernel __attribute__((amdgpu_waves_per_eu(2, 4))) void kernel_waves_per_eu_2_4() {}
kernel __attribute__((amdgpu_num_sgpr(32))) void kernel_num_sgpr_32() {}
kernel __attribute__((amdgpu_num_vgpr(64))) void kernel_num_vgpr_64() {}
