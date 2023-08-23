; Kernel test2_v3 wth printf, code-object-v3 source
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx803"
	.protected	test2_v3        ; -- Begin function test2_v3
	.globl	test2_v3
	.p2align	8
	.type	test2_v3,@function
test2_v3:                               ; @test2_v3
test2_v3$local:
; %bb.0:                                ; %entry
	s_add_u32 s4, s4, s7
	s_lshr_b32 flat_scratch_hi, s4, 8
	s_add_u32 s0, s0, s7
	s_addc_u32 s1, s1, 0
	s_mov_b32 flat_scratch_lo, s5
	s_getpc_b64 s[4:5]
	s_add_u32 s4, s4, __printf_alloc@gotpcrel32@lo+4
	s_addc_u32 s5, s5, __printf_alloc@gotpcrel32@hi+4
	s_load_dwordx2 s[4:5], s[4:5], 0x0
	v_mov_b32_e32 v0, 4
	s_mov_b32 s32, 0
	s_mov_b32 s33, 0
	s_waitcnt lgkmcnt(0)
	s_swappc_b64 s[30:31], s[4:5]
	v_cmp_ne_u64_e32 vcc, 0, v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz BB0_2
; %bb.1:
	v_mov_b32_e32 v2, 1
	flat_store_dword v[0:1], v2
BB0_2:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel test2_v3
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 16384
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 0
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 1
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 42
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	test2_v3, .Lfunc_end0-test2_v3
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 96
; NumSgprs: 48
; NumVgprs: 24
; ScratchSize: 16384
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 48
; NumVGPRsForWavesPerEU: 24
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.ident	"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 20629ca949cddde9f7e41a4b9e8539a970615feb)"
	.section	".note.GNU-stack"
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           8
        .value_kind:     hidden_global_offset_x
        .value_type:     i64
      - .offset:         8
        .size:           8
        .value_kind:     hidden_global_offset_y
        .value_type:     i64
      - .offset:         16
        .size:           8
        .value_kind:     hidden_global_offset_z
        .value_type:     i64
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     hidden_printf_buffer
        .value_type:     i8
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     hidden_none
        .value_type:     i8
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
        .value_type:     i8
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .kernarg_segment_size: 56
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           test2_v3
    .private_segment_fixed_size: 16384
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         test2_v3.kd
    .vgpr_count:     24
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.printf:
  - '1:0:foo'
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
