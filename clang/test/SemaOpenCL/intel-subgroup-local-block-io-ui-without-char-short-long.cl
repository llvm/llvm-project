// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL3.0 -fdeclare-opencl-builtins -cl-ext=+cl_intel_subgroup_local_block_io,-cl_intel_subgroups_char,-cl_intel_subgroups_short,-cl_intel_subgroups_long -verify -fsyntax-only %s

// Keep this test header-free so it exercises OpenCLBuiltins.td instead of
// declarations from opencl-c.h.
//
// Per the cl_intel_subgroup_local_block_io specification, intel_sub_group_block_read_ui*
// and intel_sub_group_block_write_ui* with __local pointer are declared by
// cl_intel_subgroup_local_block_io alone.  cl_intel_subgroups_char/short/long
// are not required and must not gate these aliases.

// expected-no-diagnostics

typedef unsigned int uint;
typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint4 __attribute__((ext_vector_type(4)));
typedef uint uint8 __attribute__((ext_vector_type(8)));

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
