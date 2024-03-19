	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
	.protected	_Z3fooPtS_              ; -- Begin function _Z3fooPtS_
	.globl	_Z3fooPtS_
	.p2align	8
	.type	_Z3fooPtS_,@function
_Z3fooPtS_:                             ; @_Z3fooPtS_
; %bb.0:
	s_clause 0x1
	s_load_dword s7, s[4:5], 0x1c
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s4, s7, 0xffff
	v_mad_u64_u32 v[0:1], null, s6, s4, v[0:1]
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	v_add_co_u32 v2, vcc_lo, s0, v0
	v_add_co_ci_u32_e32 v3, vcc_lo, s1, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, s2, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s3, v1, vcc_lo
	global_load_ushort v2, v[2:3], off
	s_waitcnt vmcnt(0)
	global_store_short v[0:1], v2, off
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z3fooPtS_
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
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
	.size	_Z3fooPtS_, .Lfunc_end0-_Z3fooPtS_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 112
; NumSgprs: 10
; NumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 4
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 6, 3214868480
	.fill 48, 4, 3214868480
	.protected	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE ; @_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE
	.type	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE
_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE, 1

	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE
_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE, 1

	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE
_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE, 1

	.ident	"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.0 23483 7208e8d15fbf218deb74483ea8c549c67ca4985e)"
	.section	".note.GNU-stack"
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z3fooPtS_
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z3fooPtS_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
