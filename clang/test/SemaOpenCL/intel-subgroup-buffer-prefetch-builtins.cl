// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL3.0 -fdeclare-opencl-builtins -verify -fsyntax-only %s

// Keep this test header-free so it exercises OpenCLBuiltins.td instead of
// declarations from opencl-c.h.

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef unsigned long ulong;

void test_block_prefetch_ui(const __global uint *in) {
  intel_sub_group_block_prefetch_ui(in);
  intel_sub_group_block_prefetch_ui2(in);
  intel_sub_group_block_prefetch_ui4(in);
  intel_sub_group_block_prefetch_ui8(in);
}

void test_block_prefetch_us(const __global ushort *in) {
  intel_sub_group_block_prefetch_us(in);
  intel_sub_group_block_prefetch_us2(in);
  intel_sub_group_block_prefetch_us4(in);
  intel_sub_group_block_prefetch_us8(in);
  intel_sub_group_block_prefetch_us16(in);
}

void test_block_prefetch_uc(const __global uchar *in) {
  intel_sub_group_block_prefetch_uc(in);
  intel_sub_group_block_prefetch_uc2(in);
  intel_sub_group_block_prefetch_uc4(in);
  intel_sub_group_block_prefetch_uc8(in);
  intel_sub_group_block_prefetch_uc16(in);
}

void test_block_prefetch_ul(const __global ulong *in) {
  intel_sub_group_block_prefetch_ul(in);
  intel_sub_group_block_prefetch_ul2(in);
  intel_sub_group_block_prefetch_ul4(in);
  intel_sub_group_block_prefetch_ul8(in);
}

void test_block_prefetch_ui16_rejected(const __global uint *in) {
  intel_sub_group_block_prefetch_ui16(in); // expected-error{{use of undeclared identifier 'intel_sub_group_block_prefetch_ui16'}}
}

void test_block_prefetch_ul16_rejected(const __global ulong *in) {
  intel_sub_group_block_prefetch_ul16(in); // expected-error{{use of undeclared identifier 'intel_sub_group_block_prefetch_ul16'}}
}

void test_block_prefetch_ui_invalid(const __global uint *in,
                                    const __local uint *local_in,
                                    const __global ushort *us_in, uint v) {
  intel_sub_group_block_prefetch_ui(); // expected-error{{too few arguments to function call, expected 1, have 0}}
  // expected-note@-1 0+{{'intel_sub_group_block_prefetch_ui' declared here}}
  intel_sub_group_block_prefetch_ui(in, in); // expected-error{{too many arguments to function call, expected 1, have 2}}
  // expected-note@-1 0+{{'intel_sub_group_block_prefetch_ui' declared here}}
  intel_sub_group_block_prefetch_ui(v); // expected-error{{incompatible integer to pointer conversion passing '__private uint'}}
  intel_sub_group_block_prefetch_ui(local_in); // expected-error{{changes address space of pointer}}
  intel_sub_group_block_prefetch_ui(us_in); // expected-error{{incompatible pointer types passing 'const __global ushort *__private'}}
}

void test_block_prefetch_us_invalid(const __global ushort *in,
                                    const __global uint *u_in, ushort v) {
  intel_sub_group_block_prefetch_us(); // expected-error{{too few arguments to function call, expected 1, have 0}}
  // expected-note@-1 0+{{'intel_sub_group_block_prefetch_us' declared here}}
  intel_sub_group_block_prefetch_us(in, in); // expected-error{{too many arguments to function call, expected 1, have 2}}
  // expected-note@-1 0+{{'intel_sub_group_block_prefetch_us' declared here}}
  intel_sub_group_block_prefetch_us(v); // expected-error{{incompatible integer to pointer conversion passing '__private ushort'}}
  intel_sub_group_block_prefetch_us(u_in); // expected-error{{incompatible pointer types passing 'const __global uint *__private'}}
}

void test_block_prefetch_uc_invalid(const __global uchar *in,
                                    const __global uint *u_in, uchar v) {
  intel_sub_group_block_prefetch_uc(); // expected-error{{too few arguments to function call, expected 1, have 0}}
  // expected-note@-1 0+{{'intel_sub_group_block_prefetch_uc' declared here}}
  intel_sub_group_block_prefetch_uc(in, in); // expected-error{{too many arguments to function call, expected 1, have 2}}
  // expected-note@-1 0+{{'intel_sub_group_block_prefetch_uc' declared here}}
  intel_sub_group_block_prefetch_uc(v); // expected-error{{incompatible integer to pointer conversion passing '__private uchar'}}
  intel_sub_group_block_prefetch_uc(u_in); // expected-error{{incompatible pointer types passing 'const __global uint *__private'}}
}

void test_block_prefetch_ul_invalid(const __global ulong *in,
                                    const __global uint *u_in, ulong v) {
  intel_sub_group_block_prefetch_ul(); // expected-error{{too few arguments to function call, expected 1, have 0}}
  // expected-note@-1 0+{{'intel_sub_group_block_prefetch_ul' declared here}}
  intel_sub_group_block_prefetch_ul(in, in); // expected-error{{too many arguments to function call, expected 1, have 2}}
  // expected-note@-1 0+{{'intel_sub_group_block_prefetch_ul' declared here}}
  intel_sub_group_block_prefetch_ul(v); // expected-error{{incompatible integer to pointer conversion passing '__private ulong'}}
  intel_sub_group_block_prefetch_ul(u_in); // expected-error{{incompatible pointer types passing 'const __global uint *__private'}}
}
