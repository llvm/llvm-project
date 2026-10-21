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
