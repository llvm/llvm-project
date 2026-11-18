// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL3.0 -fdeclare-opencl-builtins -verify -fsyntax-only %s

// Keep this test header-free so it exercises OpenCLBuiltins.td instead of
// declarations from opencl-c.h.

typedef unsigned int uint;
typedef unsigned long ulong;
typedef int int2 __attribute__((ext_vector_type(2)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef int int16 __attribute__((ext_vector_type(16)));
typedef long long2 __attribute__((ext_vector_type(2)));
typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint4 __attribute__((ext_vector_type(4)));
typedef uint uint8 __attribute__((ext_vector_type(8)));

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

float3 test_shuffle_float3(float3 value) {
  return intel_sub_group_shuffle(value, 1u);
}

int16 test_shuffle_xor_int16(int16 value) {
  return intel_sub_group_shuffle_xor(value, 1u);
}

uint8 test_shuffle_down_uint8(uint8 current, uint8 next) {
  return intel_sub_group_shuffle_down(current, next, 1u);
}

uint8 test_shuffle_up_uint8(uint8 previous, uint8 current) {
  return intel_sub_group_shuffle_up(previous, current, 1u);
}

half test_shuffle_half(half value) {
  return intel_sub_group_shuffle(value, 1u);
}

double test_shuffle_double(double value) {
  return intel_sub_group_shuffle_xor(value, 1u);
}

long test_shuffle_long(long value) {
  return intel_sub_group_shuffle(value, 1u);
}

ulong test_shuffle_ulong(ulong value) {
  return intel_sub_group_shuffle_xor(value, 1u);
}

uint test_block_read_global(const __global uint *in) {
  return intel_sub_group_block_read(in);
}

uint2 test_block_read2_global(const __global uint *in) {
  return intel_sub_group_block_read2(in);
}

uint4 test_block_read4_global(const __global uint *in) {
  return intel_sub_group_block_read4(in);
}

uint8 test_block_read8_global(const __global uint *in) {
  return intel_sub_group_block_read8(in);
}

uint test_block_read_image(read_only image2d_t image, int2 coord) {
  return intel_sub_group_block_read(image, coord);
}

uint2 test_block_read2_image(read_only image2d_t image, int2 coord) {
  return intel_sub_group_block_read2(image, coord);
}

uint4 test_block_read4_rw_image(read_write image2d_t image, int2 coord) {
  return intel_sub_group_block_read4(image, coord);
}

uint8 test_block_read8_rw_image(read_write image2d_t image, int2 coord) {
  return intel_sub_group_block_read8(image, coord);
}

void test_block_write_global(__global uint *out, uint value, uint2 value2,
                             uint4 value4, uint8 value8) {
  intel_sub_group_block_write(out, value);
  intel_sub_group_block_write2(out, value2);
  intel_sub_group_block_write4(out, value4);
  intel_sub_group_block_write8(out, value8);
}

void test_block_write_image(write_only image2d_t image, read_write image2d_t rw,
                            int2 coord, uint value, uint2 value2,
                            uint4 value4, uint8 value8) {
  intel_sub_group_block_write(image, coord, value);
  intel_sub_group_block_write2(image, coord, value2);
  intel_sub_group_block_write4(rw, coord, value4);
  intel_sub_group_block_write8(rw, coord, value8);
}

uint test_block_read_ui_global(const __global uint *in) {
  return intel_sub_group_block_read_ui(in);
}

uint2 test_block_read_ui2_global(const __global uint *in) {
  return intel_sub_group_block_read_ui2(in);
}

uint4 test_block_read_ui4_image(read_only image2d_t image, int2 coord) {
  return intel_sub_group_block_read_ui4(image, coord);
}

uint8 test_block_read_ui8_rw_image(read_write image2d_t image, int2 coord) {
  return intel_sub_group_block_read_ui8(image, coord);
}

void test_block_write_ui_global(__global uint *out, uint value, uint2 value2,
                                uint4 value4, uint8 value8) {
  intel_sub_group_block_write_ui(out, value);
  intel_sub_group_block_write_ui2(out, value2);
  intel_sub_group_block_write_ui4(out, value4);
  intel_sub_group_block_write_ui8(out, value8);
}

void test_block_write_ui_image(write_only image2d_t image,
                               read_write image2d_t rw, int2 coord,
                               uint value, uint2 value2, uint4 value4,
                               uint8 value8) {
  intel_sub_group_block_write_ui(image, coord, value);
  intel_sub_group_block_write_ui2(image, coord, value2);
  intel_sub_group_block_write_ui4(rw, coord, value4);
  intel_sub_group_block_write_ui8(rw, coord, value8);
}

void test_long_vectors_rejected(long2 value) {
  (void)intel_sub_group_shuffle(value, 0u); // expected-error{{no matching function for call to 'intel_sub_group_shuffle'}}
  // expected-note@-1 0+{{candidate function not viable}}
}

void test_shuffle_invalid(uint value, __global uint *ptr) {
  intel_sub_group_shuffle(); // expected-error{{no matching function for call to 'intel_sub_group_shuffle'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_shuffle(value); // expected-error{{no matching function for call to 'intel_sub_group_shuffle'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_shuffle(value, value, value); // expected-error{{no matching function for call to 'intel_sub_group_shuffle'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_shuffle(ptr, 1u); // expected-error{{no matching function for call to 'intel_sub_group_shuffle'}}
  // expected-note@-1 0+{{candidate function not viable}}
}

void test_shuffle_down_invalid(uint value) {
  intel_sub_group_shuffle_down(); // expected-error{{no matching function for call to 'intel_sub_group_shuffle_down'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_shuffle_down(value, value); // expected-error{{no matching function for call to 'intel_sub_group_shuffle_down'}}
  // expected-note@-1 0+{{candidate function not viable}}
}

void test_block_read_invalid(const __global uint *in, uint v) {
  intel_sub_group_block_read(); // expected-error{{no matching function for call to 'intel_sub_group_block_read'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_read(in, in); // expected-error{{no matching function for call to 'intel_sub_group_block_read'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_read(v); // expected-error{{no matching function for call to 'intel_sub_group_block_read'}}
  // expected-note@-1 0+{{candidate function not viable}}
}

void test_block_write_invalid(__global uint *out, read_only image2d_t roimg,
                              int2 coord, uint value) {
  intel_sub_group_block_write(); // expected-error{{no matching function for call to 'intel_sub_group_block_write'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write(out); // expected-error{{no matching function for call to 'intel_sub_group_block_write'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write(out, value, value); // expected-error{{no matching function for call to 'intel_sub_group_block_write'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write(roimg, coord, value); // expected-error{{no matching function for call to 'intel_sub_group_block_write'}}
  // expected-note@-1 0+{{candidate function not viable}}
}
