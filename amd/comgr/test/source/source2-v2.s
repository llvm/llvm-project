; Kernel test2_v2 wth printf, code-object-v2 source
	.text
	.hsa_code_object_version 2,1
	.hsa_code_object_isa 8,0,3,"AMD","AMDGPU"
	.protected	test2_v2        ; -- Begin function test2_v2
	.globl	test2_v2
	.p2align	8
	.type	test2_v2,@function
	.amdgpu_hsa_kernel test2_v2
test2_v2:                               ; @test2_v2
test2_v2$local:
	.amd_kernel_code_t
		amd_code_version_major = 1
		amd_code_version_minor = 2
		amd_machine_kind = 1
		amd_machine_version_major = 8
		amd_machine_version_minor = 0
		amd_machine_version_stepping = 3
		kernel_code_entry_byte_offset = 256
		kernel_code_prefetch_byte_size = 0
		granulated_workitem_vgpr_count = 5
		granulated_wavefront_sgpr_count = 5
		priority = 0
		float_mode = 192
		priv = 0
		enable_dx10_clamp = 1
		debug_mode = 0
		enable_ieee_mode = 1
		enable_wgp_mode = 0
		enable_mem_ordered = 0
		enable_fwd_progress = 0
		enable_sgpr_private_segment_wave_byte_offset = 1
		user_sgpr_count = 6
		enable_trap_handler = 0
		enable_sgpr_workgroup_id_x = 1
		enable_sgpr_workgroup_id_y = 0
		enable_sgpr_workgroup_id_z = 0
		enable_sgpr_workgroup_info = 0
		enable_vgpr_workitem_id = 0
		enable_exception_msb = 0
		granulated_lds_size = 0
		enable_exception = 0
		enable_sgpr_private_segment_buffer = 1
		enable_sgpr_dispatch_ptr = 0
		enable_sgpr_queue_ptr = 0
		enable_sgpr_kernarg_segment_ptr = 0
		enable_sgpr_dispatch_id = 0
		enable_sgpr_flat_scratch_init = 1
		enable_sgpr_private_segment_size = 0
		enable_sgpr_grid_workgroup_count_x = 0
		enable_sgpr_grid_workgroup_count_y = 0
		enable_sgpr_grid_workgroup_count_z = 0
		enable_wavefront_size32 = 0
		enable_ordered_append_gds = 0
		private_element_size = 1
		is_ptr64 = 1
		is_dynamic_callstack = 1
		is_debug_enabled = 0
		is_xnack_enabled = 0
		workitem_private_segment_byte_size = 16384
		workgroup_group_segment_byte_size = 0
		gds_segment_byte_size = 0
		kernarg_segment_byte_size = 56
		workgroup_fbarrier_count = 0
		wavefront_sgpr_count = 48
		workitem_vgpr_count = 24
		reserved_vgpr_first = 0
		reserved_vgpr_count = 0
		reserved_sgpr_first = 0
		reserved_sgpr_count = 0
		debug_wavefront_private_segment_offset_sgpr = 0
		debug_private_segment_buffer_sgpr = 0
		kernarg_segment_alignment = 4
		group_segment_alignment = 4
		private_segment_alignment = 4
		wavefront_size = 6
		call_convention = -1
		runtime_loader_kernel_symbol = 0
	.end_amd_kernel_code_t
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
.Lfunc_end0:
	.size	test2_v2, .Lfunc_end0-test2_v2
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
	.amd_amdgpu_isa "amdgcn-amd-amdhsa--gfx803"
	.amd_amdgpu_hsa_metadata
---
Version:         [ 1, 0 ]
Printf:
  - '1:0:foo'
Kernels:
  - Name:            test2_v2
    SymbolName:      'test2_v2@kd'
    Language:        OpenCL C
    LanguageVersion: [ 2, 0 ]
    Args:
      - Size:            8
        Align:           8
        ValueKind:       HiddenGlobalOffsetX
        ValueType:       I64
      - Size:            8
        Align:           8
        ValueKind:       HiddenGlobalOffsetY
        ValueType:       I64
      - Size:            8
        Align:           8
        ValueKind:       HiddenGlobalOffsetZ
        ValueType:       I64
      - Size:            8
        Align:           8
        ValueKind:       HiddenPrintfBuffer
        ValueType:       I8
        AddrSpaceQual:   Global
      - Size:            8
        Align:           8
        ValueKind:       HiddenNone
        ValueType:       I8
        AddrSpaceQual:   Global
      - Size:            8
        Align:           8
        ValueKind:       HiddenNone
        ValueType:       I8
        AddrSpaceQual:   Global
      - Size:            8
        Align:           8
        ValueKind:       HiddenMultiGridSyncArg
        ValueType:       I8
        AddrSpaceQual:   Global
    CodeProps:
      KernargSegmentSize: 56
      GroupSegmentFixedSize: 0
      PrivateSegmentFixedSize: 16384
      KernargSegmentAlign: 4
      WavefrontSize:   64
      NumSGPRs:        48
      NumVGPRs:        24
      MaxFlatWorkGroupSize: 256
      IsDynamicCallStack: true
...

	.end_amd_amdgpu_hsa_metadata
