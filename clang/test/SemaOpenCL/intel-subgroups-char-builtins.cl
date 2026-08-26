// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL3.0 -fdeclare-opencl-builtins -verify -fsyntax-only %s

// Keep this test header-free so it exercises OpenCLBuiltins.td instead of
// declarations from opencl-c.h.

typedef unsigned int uint;
typedef unsigned char uchar;
typedef int int2 __attribute__((ext_vector_type(2)));
typedef char char3 __attribute__((ext_vector_type(3)));
typedef char char8 __attribute__((ext_vector_type(8)));
typedef char char16 __attribute__((ext_vector_type(16)));
typedef uchar uchar2 __attribute__((ext_vector_type(2)));
typedef uchar uchar4 __attribute__((ext_vector_type(4)));
typedef uchar uchar8 __attribute__((ext_vector_type(8)));
typedef uchar uchar16 __attribute__((ext_vector_type(16)));

char3 test_broadcast_char3(char3 value) {
  return intel_sub_group_broadcast(value, 1u);
}

uchar8 test_broadcast_uchar8(uchar8 value) {
  return intel_sub_group_broadcast(value, 1u);
}

char16 test_shuffle_char16(char16 value) {
  return intel_sub_group_shuffle(value, 1u);
}

uchar16 test_shuffle_xor_uchar16(uchar16 value) {
  return intel_sub_group_shuffle_xor(value, 1u);
}

char16 test_shuffle_down_char16(char16 current, char16 next) {
  return intel_sub_group_shuffle_down(current, next, 1u);
}

uchar16 test_shuffle_up_uchar16(uchar16 previous, uchar16 current) {
  return intel_sub_group_shuffle_up(previous, current, 1u);
}

char test_collectives_char(char value) {
  value = intel_sub_group_reduce_add(value);
  value = intel_sub_group_reduce_min(value);
  value = intel_sub_group_reduce_max(value);
  value = intel_sub_group_scan_exclusive_add(value);
  value = intel_sub_group_scan_exclusive_min(value);
  value = intel_sub_group_scan_exclusive_max(value);
  value = intel_sub_group_scan_inclusive_add(value);
  value = intel_sub_group_scan_inclusive_min(value);
  value = intel_sub_group_scan_inclusive_max(value);
  return value;
}

uchar test_collectives_uchar(uchar value) {
  value = intel_sub_group_reduce_add(value);
  value = intel_sub_group_reduce_min(value);
  value = intel_sub_group_reduce_max(value);
  value = intel_sub_group_scan_exclusive_add(value);
  value = intel_sub_group_scan_exclusive_min(value);
  value = intel_sub_group_scan_exclusive_max(value);
  value = intel_sub_group_scan_inclusive_add(value);
  value = intel_sub_group_scan_inclusive_min(value);
  value = intel_sub_group_scan_inclusive_max(value);
  return value;
}

uchar test_block_read_uc_global(const __global uchar *in) {
  return intel_sub_group_block_read_uc(in);
}

uchar2 test_block_read_uc2_global(const __global uchar *in) {
  return intel_sub_group_block_read_uc2(in);
}

uchar4 test_block_read_uc4_image(read_only image2d_t image, int2 coord) {
  return intel_sub_group_block_read_uc4(image, coord);
}

uchar8 test_block_read_uc8_rw_image(read_write image2d_t image, int2 coord) {
  return intel_sub_group_block_read_uc8(image, coord);
}

uchar16 test_block_read_uc16_rw_image(read_write image2d_t image, int2 coord) {
  return intel_sub_group_block_read_uc16(image, coord);
}

void test_block_write_uc(__global uchar *out, write_only image2d_t image,
                         read_write image2d_t rw, int2 coord, uchar value,
                         uchar2 value2, uchar4 value4, uchar8 value8,
                         uchar16 value16) {
  intel_sub_group_block_write_uc(out, value);
  intel_sub_group_block_write_uc2(out, value2);
  intel_sub_group_block_write_uc4(out, value4);
  intel_sub_group_block_write_uc8(out, value8);
  intel_sub_group_block_write_uc16(out, value16);
  intel_sub_group_block_write_uc(image, coord, value);
  intel_sub_group_block_write_uc2(image, coord, value2);
  intel_sub_group_block_write_uc4(image, coord, value4);
  intel_sub_group_block_write_uc8(rw, coord, value8);
  intel_sub_group_block_write_uc16(rw, coord, value16);
}

void test_broadcast_char16_rejected(char16 value) {
  (void)intel_sub_group_broadcast(value, 0u); // expected-error{{no matching function for call to 'intel_sub_group_broadcast'}}
  // expected-note@-1 0+{{candidate function not viable}}
}

void test_broadcast_invalid(uchar value, __global uchar *ptr) {
  intel_sub_group_broadcast(); // expected-error{{no matching function for call to 'intel_sub_group_broadcast'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_broadcast(value); // expected-error{{no matching function for call to 'intel_sub_group_broadcast'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_broadcast(value, value, value); // expected-error{{no matching function for call to 'intel_sub_group_broadcast'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_broadcast(ptr, 1u); // expected-error{{no matching function for call to 'intel_sub_group_broadcast'}}
  // expected-note@-1 0+{{candidate function not viable}}
}

void test_block_read_uc_invalid(const __global uchar *in, uchar v) {
  intel_sub_group_block_read_uc(); // expected-error{{no matching function for call to 'intel_sub_group_block_read_uc'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_read_uc(in, in); // expected-error{{no matching function for call to 'intel_sub_group_block_read_uc'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_read_uc(v); // expected-error{{no matching function for call to 'intel_sub_group_block_read_uc'}}
  // expected-note@-1 0+{{candidate function not viable}}
}

void test_block_write_uc_invalid(__global uchar *out, read_only image2d_t roimg,
                                 int2 coord, uchar value) {
  intel_sub_group_block_write_uc(); // expected-error{{no matching function for call to 'intel_sub_group_block_write_uc'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write_uc(out); // expected-error{{no matching function for call to 'intel_sub_group_block_write_uc'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write_uc(out, value, value); // expected-error{{no matching function for call to 'intel_sub_group_block_write_uc'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write_uc(roimg, coord, value); // expected-error{{no matching function for call to 'intel_sub_group_block_write_uc'}}
  // expected-note@-1 0+{{candidate function not viable}}
}
