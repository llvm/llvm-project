// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=CL1.2 -cl-ext=+cl_intel_device_side_avc_motion_estimation -fsyntax-only -verify -DEXT %s
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=CL1.2 -cl-ext=-cl_intel_device_side_avc_motion_estimation -fsyntax-only -verify  %s

#ifdef cl_intel_device_side_avc_motion_estimation
#pragma OPENCL EXTENSION cl_intel_device_side_avc_motion_estimation : enable
#endif

// All intel_sub_group_avc_* types can only be used as argument or return value
// of built-in functions defined in the extension.
// But there are also additional initialization rules:
//   * All types except intel_sub_group_avc_mce_* types can be initialized with
//     the corresponding initializer macro defined in opencl-c.h
//     Currently all these macroses are defined as 0x0
//   * In previous versions of the extension these macroses was defined as {0},
//     so initialization with initializer list containing one integer equal to
//     zero should also work

struct st{};
// negative test cases for initializers
void foo(char c, float f, void* v, struct st ss) {
  intel_sub_group_avc_mce_payload_t payload_mce = 0; // No zero initializer for mce types
  intel_sub_group_avc_ime_payload_t payload_ime = 1; // No literal initializer for *payload_t types
  intel_sub_group_avc_ref_payload_t payload_ref = f;
  intel_sub_group_avc_sic_payload_t payload_sic = ss;
  intel_sub_group_avc_mce_result_t result_mce = 0; // No zero initializer for mce types
  intel_sub_group_avc_ime_result_t result_ime = 1; // No literal initializer for *result_t types
  intel_sub_group_avc_ref_result_t result_ref = f;
  intel_sub_group_avc_sic_result_t result_sic = ss;
  intel_sub_group_avc_ime_result_single_reference_streamout_t sstreamout = v;
  intel_sub_group_avc_ime_result_dual_reference_streamout_t dstreamin_list = {0x0, 0x1};
  intel_sub_group_avc_ime_dual_reference_streamin_t dstreamin_list2 = {};
  intel_sub_group_avc_ime_single_reference_streamin_t dstreamin_list3 = {c};
  intel_sub_group_avc_ime_dual_reference_streamin_t dstreamin_list4 = {1};
#ifdef EXT
// expected-error@-14 {{initializing '__private intel_sub_group_avc_mce_payload_t' with an expression of incompatible type 'int'}}
// expected-error@-14 {{initializing '__private intel_sub_group_avc_ime_payload_t' with an expression of incompatible type 'int'}}
// expected-error@-14 {{initializing '__private intel_sub_group_avc_ref_payload_t' with an expression of incompatible type '__private float'}}
// expected-error@-14 {{initializing '__private intel_sub_group_avc_sic_payload_t' with an expression of incompatible type '__private struct st'}}
// expected-error@-14 {{initializing '__private intel_sub_group_avc_mce_result_t' with an expression of incompatible type 'int'}}
// expected-error@-14 {{initializing '__private intel_sub_group_avc_ime_result_t' with an expression of incompatible type 'int'}}
// expected-error@-14 {{initializing '__private intel_sub_group_avc_ref_result_t' with an expression of incompatible type '__private float'}}
// expected-error@-14 {{initializing '__private intel_sub_group_avc_sic_result_t' with an expression of incompatible type '__private struct st'}}
// expected-error@-14 {{initializing '__private intel_sub_group_avc_ime_result_single_reference_streamout_t' with an expression of incompatible type '__private void *__private'}}
// expected-warning@-14 {{excess elements in struct initializer}}
// expected-error@-14 {{scalar initializer cannot be empty}}
// expected-error@-14 {{initializing '__private intel_sub_group_avc_ime_single_reference_streamin_t' with an expression of incompatible type '__private char'}}
// expected-error@-14 {{initializing '__private intel_sub_group_avc_ime_dual_reference_streamin_t' with an expression of incompatible type 'int'}}
#else
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_mce_payload_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_ime_payload_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_ref_payload_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_sic_payload_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_mce_result_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_ime_result_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_ref_result_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_sic_result_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_ime_result_single_reference_streamout_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_ime_result_dual_reference_streamout_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_ime_dual_reference_streamin_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_ime_single_reference_streamin_t'}}
// expected-error@-28 {{use of undeclared identifier 'intel_sub_group_avc_ime_dual_reference_streamin_t'}}
#endif
}

// negative tests for initializers and assignment
void far() {
  intel_sub_group_avc_mce_payload_t payload_mce;
  intel_sub_group_avc_mce_payload_t payload_mce2 = payload_mce;
  intel_sub_group_avc_ime_payload_t payload_ime;
  intel_sub_group_avc_ref_payload_t payload_ref = payload_ime;
  intel_sub_group_avc_sic_result_t result_sic;
  intel_sub_group_avc_ime_result_t result_ime;
  result_sic = result_ime;
#ifdef EXT
// expected-error@-5 {{initializing '__private intel_sub_group_avc_ref_payload_t' with an expression of incompatible type '__private intel_sub_group_avc_ime_payload_t'}}
// expected-error@-3 {{assigning to '__private intel_sub_group_avc_sic_result_t' from incompatible type '__private intel_sub_group_avc_ime_result_t'}}
#else
// expected-error@-11 {{use of undeclared identifier 'intel_sub_group_avc_mce_payload_t'}}
// expected-error@-11 {{use of undeclared identifier 'intel_sub_group_avc_mce_payload_t'}}
// expected-error@-11 {{use of undeclared identifier 'intel_sub_group_avc_ime_payload_t'}}
// expected-error@-11 {{use of undeclared identifier 'intel_sub_group_avc_ref_payload_t'}}
// expected-error@-11 {{use of undeclared identifier 'intel_sub_group_avc_sic_result_t'}}
// expected-error@-11 {{use of undeclared identifier 'intel_sub_group_avc_ime_result_t'}}
// expected-error@-11 {{use of undeclared identifier 'result_sic'}} expected-error@-11 {{use of undeclared identifier 'result_ime'}}
#endif
}

// Using 0x0 directly allows us not to include opencl-c.h header and not to
// redefine all of these CLK_AVC_*_INTITIALIZE_INTEL macro. '0x0' value must
// be in sync with ones defined in opencl-c.h

#ifdef EXT
// positive test cases
void bar() {
  const sampler_t vme_sampler = 0x0;

  intel_sub_group_avc_mce_payload_t payload_mce; // No literal initializer for mce types
  intel_sub_group_avc_ime_payload_t payload_ime = 0x0;
  intel_sub_group_avc_ref_payload_t payload_ref = 0x0;
  intel_sub_group_avc_sic_payload_t payload_sic = 0x0;

  intel_sub_group_avc_mce_result_t result_mce; // No literal initializer for mce types
  intel_sub_group_avc_ime_result_t result_ime = 0x0;
  intel_sub_group_avc_ref_result_t result_ref = 0x0;
  intel_sub_group_avc_sic_result_t result_sic = 0x0;

  intel_sub_group_avc_ime_result_single_reference_streamout_t sstreamout = 0x0;
  intel_sub_group_avc_ime_result_dual_reference_streamout_t dstreamout = 0x0;
  intel_sub_group_avc_ime_single_reference_streamin_t sstreamin = 0x0;
  intel_sub_group_avc_ime_dual_reference_streamin_t dstreamin = 0x0;

  // It is allowed to assign variables of the same types
  intel_sub_group_avc_mce_payload_t pauload_mce2 = payload_mce;

  // Initialization with initializer list was supported in the first version
  // of the extension. So we check for backward compatibility here.
  intel_sub_group_avc_ime_payload_t payload_ime_list = {0};
  intel_sub_group_avc_ref_payload_t payload_ref_list = {0};
  intel_sub_group_avc_sic_payload_t payload_sic_list = {0};

  intel_sub_group_avc_ime_result_t result_ime_list = {0};
  intel_sub_group_avc_ref_result_t result_ref_list = {0};
  intel_sub_group_avc_sic_result_t result_sic_list = {0};

  intel_sub_group_avc_ime_result_single_reference_streamout_t sstreamout_list = {0};
  intel_sub_group_avc_ime_result_dual_reference_streamout_t dstreamout_list = {0};
  intel_sub_group_avc_ime_single_reference_streamin_t sstreamin_list = {0};
  intel_sub_group_avc_ime_dual_reference_streamin_t dstreamin_list = {0};
}
#endif //EXT
