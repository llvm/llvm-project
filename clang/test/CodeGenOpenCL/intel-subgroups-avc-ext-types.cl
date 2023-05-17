// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=CL1.2 -cl-ext=+cl_intel_device_side_avc_motion_estimation -emit-llvm -o - -O0 | FileCheck %s

// CHECK: store target("spirv.AvcImePayloadINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcRefPayloadINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcSicPayloadINTEL") zeroinitializer,

// CHECK: store target("spirv.AvcImeResultINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcRefResultINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcSicResultINTEL") zeroinitializer,

// CHECK: store target("spirv.AvcImeResultSingleReferenceStreamoutINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcImeResultDualReferenceStreamoutINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcImeSingleReferenceStreaminINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcImeDualReferenceStreaminINTEL") zeroinitializer,

// CHECK: store target("spirv.AvcImePayloadINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcRefPayloadINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcSicPayloadINTEL") zeroinitializer,

// CHECK: store target("spirv.AvcImeResultINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcRefResultINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcSicResultINTEL") zeroinitializer,

// CHECK: store target("spirv.AvcImeResultSingleReferenceStreamoutINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcImeResultDualReferenceStreamoutINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcImeSingleReferenceStreaminINTEL") zeroinitializer,
// CHECK: store target("spirv.AvcImeDualReferenceStreaminINTEL") zeroinitializer,

#pragma OPENCL EXTENSION cl_intel_device_side_avc_motion_estimation : enable

// Using 0x0 directly allows us not to include opencl-c.h header and not to
// redefine all of these CLK_AVC_*_INTITIALIZE_INTEL macro. '0x0' value must
// be in sync with ones defined in opencl-c.h

void foo() {
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

