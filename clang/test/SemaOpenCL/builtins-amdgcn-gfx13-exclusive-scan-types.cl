// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu13.10-amd-amdhsa -Wsign-conversion -verify -fsyntax-only %s

typedef unsigned int uint;
typedef unsigned short ushort;

void test_exclusive_scan_sum_u32_types(int *out_i32, uint *out_u32, int src_i32, uint src_u32) {
  // unsigned return assigned to signed should warn
  *out_i32 = __builtin_amdgcn_exclusive_scan_sum_u32(src_u32, src_u32, true); // expected-warning {{implicit conversion changes signedness: 'unsigned int' to 'int'}}
  // signed args passed to unsigned params should warn
  *out_u32 = __builtin_amdgcn_exclusive_scan_sum_u32(src_i32, src_u32, true); // expected-warning {{implicit conversion changes signedness: 'int' to 'unsigned int'}}
  *out_u32 = __builtin_amdgcn_exclusive_scan_sum_u32(src_u32, src_i32, true); // expected-warning {{implicit conversion changes signedness: 'int' to 'unsigned int'}}
  // correct usage: no warnings
  *out_u32 = __builtin_amdgcn_exclusive_scan_sum_u32(src_u32, src_u32, true);
}

void test_exclusive_scan_min_u32_types(int *out_i32, uint *out_u32, int src_i32, uint src_u32) {
  *out_i32 = __builtin_amdgcn_exclusive_scan_min_u32(src_u32, src_u32); // expected-warning {{implicit conversion changes signedness: 'unsigned int' to 'int'}}
  *out_u32 = __builtin_amdgcn_exclusive_scan_min_u32(src_i32, src_u32); // expected-warning {{implicit conversion changes signedness: 'int' to 'unsigned int'}}
  *out_u32 = __builtin_amdgcn_exclusive_scan_min_u32(src_u32, src_i32); // expected-warning {{implicit conversion changes signedness: 'int' to 'unsigned int'}}
  *out_u32 = __builtin_amdgcn_exclusive_scan_min_u32(src_u32, src_u32);
}

void test_exclusive_scan_max_u32_types(int *out_i32, uint *out_u32, int src_i32, uint src_u32) {
  *out_i32 = __builtin_amdgcn_exclusive_scan_max_u32(src_u32, src_u32); // expected-warning {{implicit conversion changes signedness: 'unsigned int' to 'int'}}
  *out_u32 = __builtin_amdgcn_exclusive_scan_max_u32(src_i32, src_u32); // expected-warning {{implicit conversion changes signedness: 'int' to 'unsigned int'}}
  *out_u32 = __builtin_amdgcn_exclusive_scan_max_u32(src_u32, src_i32); // expected-warning {{implicit conversion changes signedness: 'int' to 'unsigned int'}}
  *out_u32 = __builtin_amdgcn_exclusive_scan_max_u32(src_u32, src_u32);
}

void test_exclusive_scan_min_u16_types(short *out_i16, ushort *out_u16, short src_i16, ushort src_u16) {
  *out_i16 = __builtin_amdgcn_exclusive_scan_min_u16(src_u16, 0); // expected-warning {{implicit conversion changes signedness: 'unsigned short' to 'short'}}
  *out_u16 = __builtin_amdgcn_exclusive_scan_min_u16(src_i16, 0); // expected-warning {{implicit conversion changes signedness: 'short' to 'unsigned short'}}
  *out_u16 = __builtin_amdgcn_exclusive_scan_min_u16(src_u16, 0);
}

void test_exclusive_scan_max_u16_types(short *out_i16, ushort *out_u16, short src_i16, ushort src_u16) {
  *out_i16 = __builtin_amdgcn_exclusive_scan_max_u16(src_u16, 0); // expected-warning {{implicit conversion changes signedness: 'unsigned short' to 'short'}}
  *out_u16 = __builtin_amdgcn_exclusive_scan_max_u16(src_i16, 0); // expected-warning {{implicit conversion changes signedness: 'short' to 'unsigned short'}}
  *out_u16 = __builtin_amdgcn_exclusive_scan_max_u16(src_u16, 0);
}
