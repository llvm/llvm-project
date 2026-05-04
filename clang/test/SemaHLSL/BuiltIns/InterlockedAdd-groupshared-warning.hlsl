// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only \
// RUN:   -disable-llvm-passes -verify

// Verify that calling InterlockedAdd with a groupshared destination produces
// the diagnostic about HLSL inout + groupshared. The warning fires per
// candidate overload considered during resolution.

groupshared int      gs_i32;
groupshared uint     gs_u32;
groupshared int64_t  gs_i64;
groupshared uint64_t gs_u64;

void test_2arg_int(int v) {
  // expected-warning@+1 4 {{passing groupshared variable to a parameter annotated with inout}}
  InterlockedAdd(gs_i32, v);
}

void test_2arg_uint(uint v) {
  // expected-warning@+1 4 {{passing groupshared variable to a parameter annotated with inout}}
  InterlockedAdd(gs_u32, v);
}

void test_2arg_i64(int64_t v) {
  // expected-warning@+1 4 {{passing groupshared variable to a parameter annotated with inout}}
  InterlockedAdd(gs_i64, v);
}

void test_2arg_u64(uint64_t v) {
  // expected-warning@+1 4 {{passing groupshared variable to a parameter annotated with inout}}
  InterlockedAdd(gs_u64, v);
}

void test_3arg_int(int v, out int orig) {
  // expected-warning@+1 4 {{passing groupshared variable to a parameter annotated with inout}}
  InterlockedAdd(gs_i32, v, orig);
}

void test_3arg_uint(uint v, out uint orig) {
  // expected-warning@+1 4 {{passing groupshared variable to a parameter annotated with inout}}
  InterlockedAdd(gs_u32, v, orig);
}
