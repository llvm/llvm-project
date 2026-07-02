// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL3.0 -fdeclare-opencl-builtins -verify -fsyntax-only %s

// Keep this test header-free so it exercises OpenCLBuiltins.td instead of
// declarations from opencl-c.h.

typedef unsigned int uint;
typedef unsigned long ulong;
typedef int int2 __attribute__((ext_vector_type(2)));
typedef ulong ulong2 __attribute__((ext_vector_type(2)));
typedef ulong ulong4 __attribute__((ext_vector_type(4)));
typedef ulong ulong8 __attribute__((ext_vector_type(8)));
typedef ulong ulong16 __attribute__((ext_vector_type(16)));

long test_shuffle_long(long value) {
  value = intel_sub_group_shuffle(value, 1u);
  value = intel_sub_group_shuffle_xor(value, 1u);
  value = intel_sub_group_shuffle_down(value, value, 1u);
  value = intel_sub_group_shuffle_up(value, value, 1u);
  return value;
}

ulong test_shuffle_ulong(ulong value) {
  value = intel_sub_group_shuffle(value, 1u);
  value = intel_sub_group_shuffle_xor(value, 1u);
  value = intel_sub_group_shuffle_down(value, value, 1u);
  value = intel_sub_group_shuffle_up(value, value, 1u);
  return value;
}

ulong test_block_read_ul_global(const __global ulong *in) {
  return intel_sub_group_block_read_ul(in);
}

ulong2 test_block_read_ul2_global(const __global ulong *in) {
  return intel_sub_group_block_read_ul2(in);
}

ulong4 test_block_read_ul4_image(read_only image2d_t image, int2 coord) {
  return intel_sub_group_block_read_ul4(image, coord);
}

ulong8 test_block_read_ul8_rw_image(read_write image2d_t image, int2 coord) {
  return intel_sub_group_block_read_ul8(image, coord);
}

ulong16 test_block_read_ul16_rw_image(read_write image2d_t image, int2 coord) {
  return intel_sub_group_block_read_ul16(image, coord);
}

void test_block_write_ul(__global ulong *out, write_only image2d_t image,
                         read_write image2d_t rw, int2 coord, ulong value,
                         ulong2 value2, ulong4 value4, ulong8 value8,
                         ulong16 value16) {
  intel_sub_group_block_write_ul(out, value);
  intel_sub_group_block_write_ul2(out, value2);
  intel_sub_group_block_write_ul4(out, value4);
  intel_sub_group_block_write_ul8(out, value8);
  intel_sub_group_block_write_ul(image, coord, value);
  intel_sub_group_block_write_ul2(image, coord, value2);
  intel_sub_group_block_write_ul4(image, coord, value4);
  intel_sub_group_block_write_ul8(rw, coord, value8);
  intel_sub_group_block_write_ul16(rw, coord, value16);
}

void test_block_read_ul_invalid(const __global ulong *in, ulong v) {
  intel_sub_group_block_read_ul(); // expected-error{{no matching function for call to 'intel_sub_group_block_read_ul'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_read_ul(in, in); // expected-error{{no matching function for call to 'intel_sub_group_block_read_ul'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_read_ul(v); // expected-error{{no matching function for call to 'intel_sub_group_block_read_ul'}}
  // expected-note@-1 0+{{candidate function not viable}}
}

void test_block_write_ul_invalid(__global ulong *out, read_only image2d_t roimg,
                                 int2 coord, ulong value) {
  intel_sub_group_block_write_ul(); // expected-error{{no matching function for call to 'intel_sub_group_block_write_ul'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write_ul(out); // expected-error{{no matching function for call to 'intel_sub_group_block_write_ul'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write_ul(out, value, value); // expected-error{{no matching function for call to 'intel_sub_group_block_write_ul'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write_ul(roimg, coord, value); // expected-error{{no matching function for call to 'intel_sub_group_block_write_ul'}}
  // expected-note@-1 0+{{candidate function not viable}}
}
