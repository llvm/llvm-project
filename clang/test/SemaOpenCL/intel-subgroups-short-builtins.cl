// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL3.0 -fdeclare-opencl-builtins -verify -fsyntax-only %s

// Keep this test header-free so it exercises OpenCLBuiltins.td instead of
// declarations from opencl-c.h.

typedef unsigned int uint;
typedef unsigned short ushort;
typedef int int2 __attribute__((ext_vector_type(2)));
typedef short short3 __attribute__((ext_vector_type(3)));
typedef short short8 __attribute__((ext_vector_type(8)));
typedef short short16 __attribute__((ext_vector_type(16)));
typedef ushort ushort2 __attribute__((ext_vector_type(2)));
typedef ushort ushort4 __attribute__((ext_vector_type(4)));
typedef ushort ushort8 __attribute__((ext_vector_type(8)));
typedef ushort ushort16 __attribute__((ext_vector_type(16)));

short3 test_broadcast_short3(short3 value) {
  return intel_sub_group_broadcast(value, 1u);
}

ushort8 test_broadcast_ushort8(ushort8 value) {
  return intel_sub_group_broadcast(value, 1u);
}

short16 test_shuffle_short16(short16 value) {
  return intel_sub_group_shuffle(value, 1u);
}

ushort16 test_shuffle_xor_ushort16(ushort16 value) {
  return intel_sub_group_shuffle_xor(value, 1u);
}

short16 test_shuffle_down_short16(short16 current, short16 next) {
  return intel_sub_group_shuffle_down(current, next, 1u);
}

ushort16 test_shuffle_up_ushort16(ushort16 previous, ushort16 current) {
  return intel_sub_group_shuffle_up(previous, current, 1u);
}

short test_collectives_short(short value) {
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

ushort test_collectives_ushort(ushort value) {
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

ushort test_block_read_us_global(const __global ushort *in) {
  return intel_sub_group_block_read_us(in);
}

ushort2 test_block_read_us2_global(const __global ushort *in) {
  return intel_sub_group_block_read_us2(in);
}

ushort4 test_block_read_us4_image(read_only image2d_t image, int2 coord) {
  return intel_sub_group_block_read_us4(image, coord);
}

ushort8 test_block_read_us8_rw_image(read_write image2d_t image, int2 coord) {
  return intel_sub_group_block_read_us8(image, coord);
}

ushort16 test_block_read_us16_rw_image(read_write image2d_t image, int2 coord) {
  return intel_sub_group_block_read_us16(image, coord);
}

void test_block_write_us(__global ushort *out, write_only image2d_t image,
                         read_write image2d_t rw, int2 coord, ushort value,
                         ushort2 value2, ushort4 value4, ushort8 value8,
                         ushort16 value16) {
  intel_sub_group_block_write_us(out, value);
  intel_sub_group_block_write_us2(out, value2);
  intel_sub_group_block_write_us4(out, value4);
  intel_sub_group_block_write_us8(out, value8);
  intel_sub_group_block_write_us16(out, value16);
  intel_sub_group_block_write_us(image, coord, value);
  intel_sub_group_block_write_us2(image, coord, value2);
  intel_sub_group_block_write_us4(image, coord, value4);
  intel_sub_group_block_write_us8(rw, coord, value8);
  intel_sub_group_block_write_us16(rw, coord, value16);
}

void test_broadcast_short16_rejected(short16 value) {
  (void)intel_sub_group_broadcast(value, 0u); // expected-error{{no matching function for call to 'intel_sub_group_broadcast'}}
  // expected-note@-1 0+{{candidate function not viable}}
}

void test_broadcast_invalid(ushort value, __global ushort *ptr) {
  intel_sub_group_broadcast(); // expected-error{{no matching function for call to 'intel_sub_group_broadcast'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_broadcast(value); // expected-error{{no matching function for call to 'intel_sub_group_broadcast'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_broadcast(value, value, value); // expected-error{{no matching function for call to 'intel_sub_group_broadcast'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_broadcast(ptr, 1u); // expected-error{{no matching function for call to 'intel_sub_group_broadcast'}}
  // expected-note@-1 0+{{candidate function not viable}}
}

void test_block_read_us_invalid(const __global ushort *in, ushort v) {
  intel_sub_group_block_read_us(); // expected-error{{no matching function for call to 'intel_sub_group_block_read_us'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_read_us(in, in); // expected-error{{no matching function for call to 'intel_sub_group_block_read_us'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_read_us(v); // expected-error{{no matching function for call to 'intel_sub_group_block_read_us'}}
  // expected-note@-1 0+{{candidate function not viable}}
}

void test_block_write_us_invalid(__global ushort *out, read_only image2d_t roimg,
                                 int2 coord, ushort value) {
  intel_sub_group_block_write_us(); // expected-error{{no matching function for call to 'intel_sub_group_block_write_us'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write_us(out); // expected-error{{no matching function for call to 'intel_sub_group_block_write_us'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write_us(out, value, value); // expected-error{{no matching function for call to 'intel_sub_group_block_write_us'}}
  // expected-note@-1 0+{{candidate function not viable}}
  intel_sub_group_block_write_us(roimg, coord, value); // expected-error{{no matching function for call to 'intel_sub_group_block_write_us'}}
  // expected-note@-1 0+{{candidate function not viable}}
}
