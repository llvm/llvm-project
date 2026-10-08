// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL3.0 -fdeclare-opencl-builtins -verify -fsyntax-only %s

// Keep this test header-free so it exercises OpenCLBuiltins.td instead of
// declarations from opencl-c.h.

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef unsigned long ulong;
typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint4 __attribute__((ext_vector_type(4)));
typedef uint uint8 __attribute__((ext_vector_type(8)));
typedef ushort ushort2 __attribute__((ext_vector_type(2)));
typedef ushort ushort4 __attribute__((ext_vector_type(4)));
typedef ushort ushort8 __attribute__((ext_vector_type(8)));
typedef ushort ushort16 __attribute__((ext_vector_type(16)));
typedef uchar uchar2 __attribute__((ext_vector_type(2)));
typedef uchar uchar4 __attribute__((ext_vector_type(4)));
typedef uchar uchar8 __attribute__((ext_vector_type(8)));
typedef uchar uchar16 __attribute__((ext_vector_type(16)));
typedef ulong ulong2 __attribute__((ext_vector_type(2)));
typedef ulong ulong4 __attribute__((ext_vector_type(4)));
typedef ulong ulong8 __attribute__((ext_vector_type(8)));
typedef ulong ulong16 __attribute__((ext_vector_type(16)));

uint test_block_read_local(const __local uint *in) {
  return intel_sub_group_block_read(in);
}

uint2 test_block_read2_local(const __local uint *in) {
  return intel_sub_group_block_read2(in);
}

uint4 test_block_read4_local(const __local uint *in) {
  return intel_sub_group_block_read4(in);
}

uint8 test_block_read8_local(const __local uint *in) {
  return intel_sub_group_block_read8(in);
}

void test_block_write_local(__local uint *out, uint value, uint2 value2,
                            uint4 value4, uint8 value8) {
  intel_sub_group_block_write(out, value);
  intel_sub_group_block_write2(out, value2);
  intel_sub_group_block_write4(out, value4);
  intel_sub_group_block_write8(out, value8);
}

uint test_block_read_ui_local(const __local uint *in) {
  return intel_sub_group_block_read_ui(in);
}

uint2 test_block_read_ui2_local(const __local uint *in) {
  return intel_sub_group_block_read_ui2(in);
}

uint4 test_block_read_ui4_local(const __local uint *in) {
  return intel_sub_group_block_read_ui4(in);
}

uint8 test_block_read_ui8_local(const __local uint *in) {
  return intel_sub_group_block_read_ui8(in);
}

void test_block_write_ui_local(__local uint *out, uint value, uint2 value2,
                               uint4 value4, uint8 value8) {
  intel_sub_group_block_write_ui(out, value);
  intel_sub_group_block_write_ui2(out, value2);
  intel_sub_group_block_write_ui4(out, value4);
  intel_sub_group_block_write_ui8(out, value8);
}

uchar test_block_read_uc_local(const __local uchar *in) {
  return intel_sub_group_block_read_uc(in);
}

uchar2 test_block_read_uc2_local(const __local uchar *in) {
  return intel_sub_group_block_read_uc2(in);
}

uchar4 test_block_read_uc4_local(const __local uchar *in) {
  return intel_sub_group_block_read_uc4(in);
}

uchar8 test_block_read_uc8_local(const __local uchar *in) {
  return intel_sub_group_block_read_uc8(in);
}

uchar16 test_block_read_uc16_local(const __local uchar *in) {
  return intel_sub_group_block_read_uc16(in);
}

void test_block_write_uc_local(__local uchar *out, uchar value, uchar2 value2,
                               uchar4 value4, uchar8 value8,
                               uchar16 value16) {
  intel_sub_group_block_write_uc(out, value);
  intel_sub_group_block_write_uc2(out, value2);
  intel_sub_group_block_write_uc4(out, value4);
  intel_sub_group_block_write_uc8(out, value8);
  intel_sub_group_block_write_uc16(out, value16);
}

ushort test_block_read_us_local(const __local ushort *in) {
  return intel_sub_group_block_read_us(in);
}

ushort2 test_block_read_us2_local(const __local ushort *in) {
  return intel_sub_group_block_read_us2(in);
}

ushort4 test_block_read_us4_local(const __local ushort *in) {
  return intel_sub_group_block_read_us4(in);
}

ushort8 test_block_read_us8_local(const __local ushort *in) {
  return intel_sub_group_block_read_us8(in);
}

ushort16 test_block_read_us16_local(const __local ushort *in) {
  return intel_sub_group_block_read_us16(in);
}

void test_block_write_us_local(__local ushort *out, ushort value,
                               ushort2 value2, ushort4 value4,
                               ushort8 value8, ushort16 value16) {
  intel_sub_group_block_write_us(out, value);
  intel_sub_group_block_write_us2(out, value2);
  intel_sub_group_block_write_us4(out, value4);
  intel_sub_group_block_write_us8(out, value8);
  intel_sub_group_block_write_us16(out, value16);
}

ulong test_block_read_ul_local(const __local ulong *in) {
  return intel_sub_group_block_read_ul(in);
}

ulong2 test_block_read_ul2_local(const __local ulong *in) {
  return intel_sub_group_block_read_ul2(in);
}

ulong4 test_block_read_ul4_local(const __local ulong *in) {
  return intel_sub_group_block_read_ul4(in);
}

ulong8 test_block_read_ul8_local(const __local ulong *in) {
  return intel_sub_group_block_read_ul8(in);
}

void test_block_write_ul_local(__local ulong *out, ulong value,
                               ulong2 value2, ulong4 value4, ulong8 value8) {
  intel_sub_group_block_write_ul(out, value);
  intel_sub_group_block_write_ul2(out, value2);
  intel_sub_group_block_write_ul4(out, value4);
  intel_sub_group_block_write_ul8(out, value8);
}

void test_block_read_ui16_local_rejected(const __local uint *in) {
  intel_sub_group_block_read_ui16(in); // expected-error{{use of undeclared identifier 'intel_sub_group_block_read_ui16'}}
}

void test_block_read_ul16_local_rejected(const __local ulong *in) {
  intel_sub_group_block_read_ul16(in); // expected-error{{no matching function for call to 'intel_sub_group_block_read_ul16'}}
  // expected-note@-1 0+{{candidate function not viable}}
}
