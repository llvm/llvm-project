; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu10.10-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --emit-ir=fp_mode_kernel \
; RUN:   | %FileCheck %s --check-prefix=MODE

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx1250 \
; RUN:   --emit-ir=fp_mode_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=DX10-OFF-REFUSE
; DX10-OFF-REFUSE: unsupported-floating-point-mode: v_add_f32 [VOP2]
; DX10-OFF-REFUSE-SAME: source DX10_CLAMP=0 is not representable on a target
; DX10-OFF-REFUSE-SAME: with fixed DX10 clamp mode

; RUN: %transpile_cli %t.hsaco --target-isa=gfx1250 \
; RUN:   --emit-ir=fixed_mode_kernel,integer_modes_off_kernel \
; RUN:   | %FileCheck %s --check-prefix=FIXED-TARGET

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx1250 \
; RUN:   --emit-ir=ieee_off_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=IEEE-OFF-REFUSE
; IEEE-OFF-REFUSE: unsupported-floating-point-mode: v_add_f32 [VOP2]
; IEEE-OFF-REFUSE-SAME: source IEEE_MODE=0 is not representable on a target
; IEEE-OFF-REFUSE-SAME: with fixed IEEE mode

; RUN: %transpile_cli %t.hsaco --target-isa=gfx1010 \
; RUN:   --emit-ir=ieee_off_kernel \
; RUN:   | %FileCheck %s --check-prefix=IEEE-OFF-SAME-ISA

; Unsupported rounding modes and lane-modifying encodings are refused.
; RUN: not %transpile_cli %t.hsaco \
; RUN:   --emit-ir=rounding_kernel,dpp_kernel,sdwa_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE
; REFUSE:      unsupported-floating-point-mode: v_add_f32 [VOP2]
; REFUSE-SAME: f32 rounding mode 1 is unsupported
; REFUSE:      unsupported-instruction-form: v_add_f32 [DPP]
; REFUSE:      unsupported-instruction-form: v_add_f32 [SDWA]

; RUN: %transpile_cli %t.hsaco --emit-ir=e64_kernel \
; RUN:   | %FileCheck %s --check-prefix=E64
; E64: fadd float

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx1010 \
; RUN:   --emit-ir=legacy_max,legacy_min 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=LEGACY
; LEGACY: unsupported-instruction-form: v_max_f32 [VOP2]
; LEGACY-SAME: legacy floating-point min/max semantics are not supported
; LEGACY: unsupported-instruction-form: v_min_f32 [VOP3]
; LEGACY-SAME: legacy floating-point min/max semantics are not supported

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1010"
	.amdhsa_code_object_version 6
	.text
	.globl	fp_mode_kernel
	.p2align	8
	.type	fp_mode_kernel,@function
; MODE-LABEL: define amdgpu_kernel void @fp_mode_kernel()
; MODE-SAME: #[[ATTR:[0-9]+]] {
fp_mode_kernel:
; MODE: fadd float
	v_add_f32_e32 v0, v1, v2
	s_endpgm

	.globl	fixed_mode_kernel
	.p2align	8
	.type	fixed_mode_kernel,@function
; FIXED-TARGET-LABEL: define amdgpu_kernel void @fixed_mode_kernel()
fixed_mode_kernel:
; FIXED-TARGET: fadd float
	v_add_f32_e32 v0, v1, v2
	s_endpgm

	.globl	ieee_off_kernel
	.p2align	8
	.type	ieee_off_kernel,@function
; IEEE-OFF-SAME-ISA-LABEL: define amdgpu_kernel void @ieee_off_kernel()
; IEEE-OFF-SAME-ISA-SAME: #[[IEEE_OFF_ATTR:[0-9]+]] {
ieee_off_kernel:
; IEEE-OFF-SAME-ISA: fadd float
	v_add_f32_e32 v0, v1, v2
	s_endpgm

	.globl	integer_modes_off_kernel
	.p2align	8
	.type	integer_modes_off_kernel,@function
; FIXED-TARGET-LABEL: define amdgpu_kernel void @integer_modes_off_kernel()
integer_modes_off_kernel:
	s_mov_b32 s0, 0
; FIXED-TARGET: ret void
	s_endpgm

	.globl	rounding_kernel
	.p2align	8
	.type	rounding_kernel,@function
rounding_kernel:
	v_add_f32_e32 v0, v1, v2
	s_endpgm

	.globl	e64_kernel
	.p2align	8
	.type	e64_kernel,@function
e64_kernel:
	v_add_f32_e64 v0, v1, v2
	s_endpgm

	.globl	dpp_kernel
	.p2align	8
	.type	dpp_kernel,@function
dpp_kernel:
	v_add_f32_dpp v0, v1, v2 row_shr:1
	s_endpgm

	.globl	sdwa_kernel
	.p2align	8
	.type	sdwa_kernel,@function
sdwa_kernel:
	v_add_f32_sdwa v0, v1, v2
	s_endpgm

	.globl legacy_max
	.p2align 8
	.type legacy_max,@function
legacy_max:
	v_max_f32_e32 v0, v1, v2
	s_endpgm

	.globl legacy_min
	.p2align 8
	.type legacy_min,@function
legacy_min:
	v_min_f32_e64 v0, v1, v2
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
; MODE: attributes #[[ATTR]] = {
; MODE-SAME: {{.*}}denormal_fpenv(preservesign|ieee,
; MODE-SAME: float: ieee|preservesign)
; MODE-SAME: {{.*}}"amdgpu-dx10-clamp"="false"
; MODE-SAME: {{.*}}"amdgpu-ieee"="true"
	.amdhsa_kernel fp_mode_kernel
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_denorm_mode_32 2
		.amdhsa_float_denorm_mode_16_64 1
		.amdhsa_dx10_clamp 0
		.amdhsa_ieee_mode 1
	.end_amdhsa_kernel
	.amdhsa_kernel fixed_mode_kernel
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
	.end_amdhsa_kernel
; IEEE-OFF-SAME-ISA: attributes #[[IEEE_OFF_ATTR]] = {
; IEEE-OFF-SAME-ISA-SAME: {{.*}}"amdgpu-ieee"="false"
	.amdhsa_kernel ieee_off_kernel
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 0
	.end_amdhsa_kernel
	.amdhsa_kernel integer_modes_off_kernel
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_dx10_clamp 0
		.amdhsa_ieee_mode 0
	.end_amdhsa_kernel
	.amdhsa_kernel rounding_kernel
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_round_mode_32 1
	.end_amdhsa_kernel
	.amdhsa_kernel e64_kernel
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel dpp_kernel
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel sdwa_kernel
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel legacy_max
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_ieee_mode 1
	.end_amdhsa_kernel
	.amdhsa_kernel legacy_min
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
		.amdhsa_ieee_mode 0
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           fp_mode_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         fp_mode_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           fixed_mode_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         fixed_mode_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           ieee_off_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         ieee_off_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           integer_modes_off_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         integer_modes_off_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           rounding_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         rounding_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           e64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         e64_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           dpp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         dpp_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sdwa_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sdwa_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 32
    .name: legacy_max
    .private_segment_fixed_size: 0
    .sgpr_count: 1
    .symbol: legacy_max.kd
    .vgpr_count: 3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 32
    .name: legacy_min
    .private_segment_fixed_size: 0
    .sgpr_count: 1
    .symbol: legacy_min.kd
    .vgpr_count: 3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
