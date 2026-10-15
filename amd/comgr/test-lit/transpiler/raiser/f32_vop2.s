; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=f32_vop2_kernel | %FileCheck %s --check-prefix=IR

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	f32_vop2_kernel
	.p2align	8
	.type	f32_vop2_kernel,@function
; IR-LABEL: define amdgpu_kernel void @f32_vop2_kernel()
; IR-SAME: #[[ATTR:[0-9]+]] {
f32_vop2_kernel:
; IR: [[ADD:%.+]] = fadd float 1.000000e+00, {{.+}}
	v_add_f32_e32 v0, 1.0, v1
; IR: [[ADD_BITS:%.+]] = bitcast float [[ADD]] to i32
; IR: [[ADD_QUIET:%.+]] = or i32 {{%.+}}, 4194304
; IR: [[ADD_RESULT:%.+]] = select i1 {{%.+}}, i32 [[ADD_QUIET]], i32 {{%.+}}
; IR: [[ADD_REG:%.+]] = phi i32 {{.+}}[[ADD_RESULT]]{{.+}}
; IR: [[ADD_VALUE:%.+]] = bitcast i32 [[ADD_REG]] to float
; IR: [[SUB:%.+]] = fsub float 2.000000e+00, [[ADD_VALUE]]
	v_sub_f32_e32 v2, 2.0, v0
; IR: [[SUB_BITS:%.+]] = bitcast float [[SUB]] to i32
; IR: [[SUB_QUIET:%.+]] = or i32 {{%.+}}, 4194304
; IR: [[SUB_RESULT:%.+]] = select i1 {{%.+}}, i32 [[SUB_QUIET]], i32 {{%.+}}
; IR: [[SUB_REG:%.+]] = phi i32 {{.+}}[[SUB_RESULT]]{{.+}}
; IR: [[SUB_VALUE:%.+]] = bitcast i32 [[SUB_REG]] to float
; IR: [[SUBREV:%.+]] = fsub float [[SUB_VALUE]], 2.000000e+00
	v_subrev_f32_e32 v3, 2.0, v2
; IR: [[SUBREV_BITS:%.+]] = bitcast float [[SUBREV]] to i32
; IR: [[SUBREV_QUIET:%.+]] = or i32 {{%.+}}, 4194304
; IR: [[SUBREV_RESULT:%.+]] = select i1 {{%.+}}, i32 [[SUBREV_QUIET]], i32 {{%.+}}
; IR: [[SUBREV_REG:%.+]] = phi i32 {{.+}}[[SUBREV_RESULT]]{{.+}}
; IR: [[SUBREV_VALUE:%.+]] = bitcast i32 [[SUBREV_REG]] to float
; IR: fmul float 5.000000e-01, [[SUBREV_VALUE]]
	v_mul_f32_e32 v4, 0.5, v3
; IR: ret void
	s_endpgm
; IR: attributes #[[ATTR]] = {
; IR-SAME: {{.*}}denormal_fpenv(float: preservesign|ieee)

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel f32_vop2_kernel
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_denorm_mode_32 1
		.amdhsa_float_denorm_mode_16_64 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           f32_vop2_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         f32_vop2_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
