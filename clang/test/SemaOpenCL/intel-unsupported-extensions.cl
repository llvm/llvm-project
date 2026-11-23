// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -finclude-default-header -cl-std=CL3.0 -cl-ext=-cl_intel_subgroups,-cl_intel_subgroup_buffer_prefetch,-cl_intel_subgroups_char,-cl_intel_subgroups_short,-cl_intel_subgroups_long,-cl_intel_bfloat16_conversions,-cl_intel_subgroup_local_block_io,-cl_intel_device_side_avc_motion_estimation %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -finclude-default-header -cl-std=CL3.0 %s

#if defined(cl_intel_subgroups) && defined(cl_intel_subgroup_buffer_prefetch) && defined(cl_intel_subgroups_char) && defined(cl_intel_subgroups_short) && defined(cl_intel_subgroups_long) && defined(cl_intel_bfloat16_conversions) && defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_device_side_avc_motion_estimation)
// expected-no-diagnostics
#endif

uint test1(read_only image2d_t im, int2 i) {
  return intel_sub_group_block_read(im, i);
}
#if !defined(cl_intel_subgroups)
// expected-error@-3{{use of undeclared identifier 'intel_sub_group_block_read'}}
#endif

void test2(const __global uint *p) {
  return intel_sub_group_block_prefetch_ui(p);
}
#if !defined(cl_intel_subgroup_buffer_prefetch)
// expected-error@-3{{use of undeclared identifier 'intel_sub_group_block_prefetch_ui'}}
#endif

uchar test3(read_only image2d_t im, int2 i) {
  return intel_sub_group_block_read_uc(im, i);
}
#if !defined(cl_intel_subgroups_char)
// expected-error@-3{{use of undeclared identifier 'intel_sub_group_block_read_uc'}}
#endif

ushort test4(const __local ushort* p) {
  return intel_sub_group_block_read_us(p);
}
#if !defined(cl_intel_subgroups_short)
// expected-error@-3{{use of undeclared identifier 'intel_sub_group_block_read_us'}}
#endif

ulong test5(const __global ulong* p) {
  return intel_sub_group_block_read_ul(p);
}
#if !defined(cl_intel_subgroups_long)
// expected-error@-3{{use of undeclared identifier 'intel_sub_group_block_read_ul'}}
#endif

ushort test6(float f) {
  return intel_convert_bfloat16_as_ushort(f);
}
#if !defined(cl_intel_bfloat16_conversions)
// expected-error@-3{{use of undeclared identifier 'intel_convert_bfloat16_as_ushort'}}
#endif

uint test7(const __local uint* p) {
  return intel_sub_group_block_read(p);
}
#if !defined(cl_intel_subgroup_local_block_io)
// expected-error@-3{{use of undeclared identifier 'intel_sub_group_block_read'}}
#endif

uchar test8(uchar slice_type, uchar qp) {
  return intel_sub_group_avc_mce_get_default_inter_base_multi_reference_penalty(slice_type, qp);
}
#if !defined(cl_intel_device_side_avc_motion_estimation)
// expected-error@-3{{use of undeclared identifier 'intel_sub_group_avc_mce_get_default_inter_base_multi_reference_penalty'}}
#endif
