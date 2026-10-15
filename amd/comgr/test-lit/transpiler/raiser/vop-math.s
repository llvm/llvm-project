; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vop_math,vop3_math | %FileCheck %s
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=literal_f64 | %FileCheck %s --check-prefix=LITERAL
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_clamp 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-CLAMP
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_omod 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-OMOD
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_div_scale_modifiers 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-SCALE-MODS
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=true16_destination \
; RUN:   | %FileCheck %s --check-prefix=TRUE16
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_tanh 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-TANH
; RUN: %transpile_cli %t.hsaco --target-isa=gfx1250 \
; RUN:   --emit-ir=refuse_tanh | %FileCheck %s --check-prefix=SUPPORT-TANH
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_bf16_denorm 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-BF16-DENORM
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_fmac_clamp 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-FMAC-CLAMP
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_fmac_omod 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-FMAC-OMOD
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_f64_clamp 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-F64-CLAMP
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_f64_omod 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-F64-OMOD
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=refuse_f64_rounding 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-F64-ROUNDING

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	vop_math
	.p2align	8
	.type	vop_math,@function
; CHECK-LABEL: define amdgpu_kernel void @vop_math(
vop_math:
	v_mov_b32 v50, v0
; CHECK: sitofp i32
	v_cvt_f32_i32_e32 v0, v1
; CHECK: uitofp i32
	v_cvt_f32_u32_e32 v2, v3
; CHECK: call i32 @llvm.fptosi.sat.i32.f32
	v_cvt_i32_f32_e32 v4, v5
; CHECK: call i32 @llvm.fptoui.sat.i32.f32
	v_cvt_u32_f32_e32 v6, v7
; CHECK: fpext half
	v_cvt_f32_f16_e32 v10, v11.l
; CHECK: uitofp i32
	v_cvt_f32_ubyte0_e32 v40, v41
; CHECK: lshr i32 {{.+}}, 8
; CHECK: uitofp i32
	v_cvt_f32_ubyte1_e32 v42, v43
; CHECK: lshr i32 {{.+}}, 16
; CHECK: uitofp i32
	v_cvt_f32_ubyte2_e32 v44, v45
; CHECK: lshr i32 {{.+}}, 24
; CHECK: uitofp i32
	v_cvt_f32_ubyte3_e32 v46, v47
; CHECK: call float @llvm.amdgcn.fract.f32
	v_fract_f32_e32 v12, v13
; CHECK: call float @llvm.trunc.f32
	v_trunc_f32_e32 v14, v15
; CHECK: call float @llvm.ceil.f32
	v_ceil_f32_e32 v16, v17
; CHECK: call float @llvm.roundeven.f32
	v_rndne_f32_e32 v18, v19
; CHECK: call float @llvm.floor.f32
	v_floor_f32_e32 v20, v21
; CHECK: call float @llvm.amdgcn.exp2.f32
	v_exp_f32_e32 v22, v23
; CHECK: call float @llvm.amdgcn.log.f32
	v_log_f32_e32 v24, v25
; CHECK: call float @llvm.amdgcn.rcp.f32
	v_rcp_f32_e32 v26, v27
; CHECK: call float asm sideeffect "v_rcp_iflag_f32 $0, $1", "=v,v"(float
	v_rcp_iflag_f32_e32 v26, v27
; CHECK: call float @llvm.amdgcn.rsq.f32
	v_rsq_f32_e32 v28, v29
; CHECK: call float @llvm.amdgcn.sqrt.f32
	v_sqrt_f32_e32 v30, v31
; CHECK: call float @llvm.amdgcn.sin.f32
	v_sin_f32_e32 v32, v33
; CHECK: call float @llvm.amdgcn.cos.f32
	v_cos_f32_e32 v34, v35
; CHECK: call i32 @llvm.amdgcn.frexp.exp.i32.f32
	v_frexp_exp_i32_f32_e32 v36, v37
; CHECK: call float @llvm.amdgcn.frexp.mant.f32
	v_frexp_mant_f32_e32 v38, v39
; CHECK: fptrunc double
	v_cvt_f32_f64_e32 v0, v[2:3]
; CHECK: fpext float
	v_cvt_f64_f32_e32 v[4:5], v6
; CHECK: sitofp i32 {{.+}} to double
	v_cvt_f64_i32_e32 v[4:5], v6
; CHECK: uitofp i32 {{.+}} to double
	v_cvt_f64_u32_e32 v[4:5], v6
; CHECK: call i32 @llvm.fptosi.sat.i32.f64
	v_cvt_i32_f64_e32 v0, v[2:3]
; CHECK: call i32 @llvm.fptoui.sat.i32.f64
	v_cvt_u32_f64_e32 v0, v[2:3]
	v_mov_b32 v2, v50
	v_mov_b32 v3, 0x3ff00000
; CHECK: call double @llvm.trunc.f64
	v_trunc_f64_e32 v[0:1], v[2:3]
	global_store_dwordx2 v[48:49], v[0:1], off
; CHECK: call double @llvm.ceil.f64
	v_ceil_f64_e32 v[0:1], v[2:3]
	global_store_dwordx2 v[48:49], v[0:1], off
; CHECK: call double @llvm.roundeven.f64
	v_rndne_f64_e32 v[0:1], v[2:3]
	global_store_dwordx2 v[48:49], v[0:1], off
; CHECK: call double @llvm.floor.f64
	v_floor_f64_e32 v[0:1], v[2:3]
	global_store_dwordx2 v[48:49], v[0:1], off
; CHECK: call double @llvm.amdgcn.rcp.f64
	v_rcp_f64_e32 v[0:1], v[2:3]
	global_store_dwordx2 v[48:49], v[0:1], off
; CHECK: call double @llvm.amdgcn.rsq.f64
	v_rsq_f64_e32 v[0:1], v[2:3]
	global_store_dwordx2 v[48:49], v[0:1], off
; CHECK: ret void
	s_endpgm

	.globl	vop3_math
	.p2align	8
	.type	vop3_math,@function
; CHECK-LABEL: define amdgpu_kernel void @vop3_math(
vop3_math:
	v_mov_b32 v16, v0
; CHECK: [[NEG:%.+]] = fneg float
; CHECK: call float @llvm.amdgcn.exp2.f32(float [[NEG]])
	v_exp_f32_e64 v0, -v1
; CHECK: [[ABS:%.+]] = call float @llvm.fabs.f32
; CHECK: call i32 @llvm.fptosi.sat.i32.f32(float [[ABS]])
	v_cvt_i32_f32_e64 v8, abs(v9)
; CHECK: [[HIGH:%.+]] = lshr i32 {{.+}}, 16
; CHECK: [[LOW:%.+]] = trunc i32 [[HIGH]] to i16
; CHECK: [[HALF:%.+]] = bitcast i16 [[LOW]] to half
; CHECK: fpext half [[HALF]] to float
	v_cvt_f32_f16_e64 v10, v11.h
; CHECK: call float @llvm.ldexp.f32.i32
	v_ldexp_f32 v2, v3, v4
	v_mov_b32 v6, v16
	v_add_nc_u32 v7, v16, 1
	v_mov_b32 v2, v6
	v_mov_b32 v3, 0x3ff00000
	v_mov_b32 v4, v7
	v_mov_b32 v5, 0x40000000
; CHECK: fneg double
; CHECK: call double @llvm.ldexp.f64.i32
	v_ldexp_f64 v[0:1], -v[2:3], v4
	global_store_dwordx2 v[14:15], v[0:1], off
; CHECK: fneg double
; CHECK: call double @llvm.amdgcn.rcp.f64
	v_rcp_f64_e64 v[0:1], -v[2:3]
	global_store_dwordx2 v[14:15], v[0:1], off
; CHECK: call double @llvm.fabs.f64
; CHECK: call double @llvm.roundeven.f64
	v_rndne_f64_e64 v[0:1], abs(v[2:3])
	global_store_dwordx2 v[14:15], v[0:1], off
; CHECK: sitofp i32 {{.+}} to double
	v_cvt_f64_i32_e32 v[0:1], v6
; CHECK: [[ACC64_LO_E32:%.+]] = zext i32 %Vgpr0{{[^ ,]*}} to i64
; CHECK: [[ACC64_HI_E32:%.+]] = zext i32 %Vgpr1{{[^ ,]*}} to i64
; CHECK: [[ACC64_SHIFT_E32:%.+]] = shl i64 [[ACC64_HI_E32]], 32
; CHECK: [[ACC64_BITS_E32:%.+]] = or i64 [[ACC64_LO_E32]], [[ACC64_SHIFT_E32]]
; CHECK: [[ACC64_E32:%.+]] = bitcast i64 [[ACC64_BITS_E32]] to double
; CHECK: call double @llvm.fma.f64(double {{.+}}, double {{.+}}, double [[ACC64_E32]])
	v_fmac_f64_e32 v[0:1], v[2:3], v[4:5]
	global_store_dwordx2 v[14:15], v[0:1], off
; CHECK: sitofp i32 {{.+}} to double
	v_cvt_f64_i32_e32 v[0:1], v7
; CHECK: [[NEG64:%.+]] = fneg double
; CHECK: [[ACC64_LO_E64:%.+]] = zext i32 %Vgpr0{{[^ ,]*}} to i64
; CHECK: [[ACC64_HI_E64:%.+]] = zext i32 %Vgpr1{{[^ ,]*}} to i64
; CHECK: [[ACC64_SHIFT_E64:%.+]] = shl i64 [[ACC64_HI_E64]], 32
; CHECK: [[ACC64_BITS_E64:%.+]] = or i64 [[ACC64_LO_E64]], [[ACC64_SHIFT_E64]]
; CHECK: [[ACC64_E64:%.+]] = bitcast i64 [[ACC64_BITS_E64]] to double
; CHECK: call double @llvm.fma.f64(double [[NEG64]], double {{.+}}, double [[ACC64_E64]])
	v_fmac_f64_e64 v[0:1], -v[2:3], v[4:5]
	global_store_dwordx2 v[14:15], v[0:1], off
; CHECK: call double @llvm.fma.f64(double {{.+}}, double 1.500000e+00, double {{.+}})
	v_fmamk_f64 v[0:1], v[2:3], 1.5, v[4:5]
	global_store_dwordx2 v[14:15], v[0:1], off
	v_cvt_f32_u32_e32 v3, v6
	v_cvt_f32_u32_e32 v4, v7
; CHECK: sitofp i32 {{.+}} to float
	v_cvt_f32_i32_e32 v2, v6
; CHECK: [[ACC32_E32:%.+]] = bitcast i32 %Vgpr2{{[^ ,]*}} to float
; CHECK: call float @llvm.fma.f32(float {{.+}}, float {{.+}}, float [[ACC32_E32]])
	v_fmac_f32_e32 v2, v3, v4
	global_store_dword v[14:15], v2, off
; CHECK: sitofp i32 {{.+}} to float
	v_cvt_f32_i32_e32 v2, v7
; CHECK: [[NEG32:%.+]] = fneg float
; CHECK: [[ACC32_E64:%.+]] = bitcast i32 %Vgpr2{{[^ ,]*}} to float
; CHECK: call float @llvm.fma.f32(float [[NEG32]], float {{.+}}, float [[ACC32_E64]])
	v_fmac_f32_e64 v2, -v3, v4
	global_store_dword v[14:15], v2, off
; CHECK: call float @llvm.fma.f32(float {{.+}}, float 1.500000e+00, float {{.+}})
	v_fmamk_f32 v2, v3, 1.5, v4
	global_store_dword v[14:15], v2, off
; CHECK: call float @llvm.amdgcn.exp2.f32
	v_s_exp_f32 s0, s1
; CHECK: call float @llvm.amdgcn.log.f32
	v_s_log_f32 s0, s1
; CHECK: call float @llvm.amdgcn.rcp.f32
	v_s_rcp_f32 s0, s1
; CHECK: call float @llvm.amdgcn.rsq.f32
	v_s_rsq_f32 s0, s1
; CHECK: call float @llvm.amdgcn.sqrt.f32
	v_s_sqrt_f32 s0, s1
; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32
; CHECK: [[SCALE_FLAG:%.+]] = extractvalue { float, i1 } {{%.+}}, 1
; CHECK: call i64 @llvm.amdgcn.ballot.i64(i1
	v_div_scale_f32 v0, s4, v1, v1, v2
	s_mov_b32 vcc_lo, s4
; CHECK: [[FMAS_FLAG:%.+]] = icmp ne i32 {{%.+}}, 0
; CHECK: call float @llvm.amdgcn.div.fmas.f32
; CHECK-SAME: i1 [[FMAS_FLAG]])
	v_div_fmas_f32 v3, v0, v1, v2
; CHECK: call float @llvm.amdgcn.div.fixup.f32
	v_div_fixup_f32 v4, v3, v1, v2
; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float {{.+}}, float {{.+}}, i1 true)
	v_div_scale_f32 v0, vcc_lo, v0, v1, v0
; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float {{.+}}, float {{.+}}, i1 false)
	v_div_scale_f32 v0, null, v1, v1, 1.0
; CHECK: call float @llvm.maximumnum.f32
	v_max3_num_f32 v5, v0, v1, v2
; CHECK: call float @llvm.minimumnum.f32
	v_min3_num_f32 v6, v0, v1, v2
; CHECK: call float @llvm.amdgcn.fmed3.f32
; CHECK: call float @llvm.minimumnum.f32
; CHECK: select i1 {{.+}}, float {{.+}}, float
	v_med3_num_f32 v7, v0, v1, v2
; CHECK: call float @llvm.maximum.f32
	v_maximum_f32 v8, v0, v1
; CHECK: call float @llvm.minimum.f32
	v_minimum_f32 v9, v0, v1
; CHECK: call float @llvm.maximum.f32
	v_maximum3_f32 v8, v0, v1, v2
; CHECK: call float @llvm.minimum.f32
	v_minimum3_f32 v9, v0, v1, v2
; CHECK: call float @llvm.maximum.f32
	v_minimummaximum_f32 v8, v0, v1, v2
; CHECK: call float @llvm.minimum.f32
	v_maximumminimum_f32 v9, v0, v1, v2
; CHECK: fpext bfloat {{.+}} to float
; CHECK: fpext bfloat {{.+}} to float
; CHECK: fpext bfloat {{.+}} to float
; CHECK: call float @llvm.fma.f32
	v_fma_mix_f32_bf16 v10, v11, v12, v13 op_sel_hi:[1,1,1]
; CHECK: lshr i32 {{.+}}, 16
; CHECK: fpext bfloat {{.+}} to float
; CHECK: bitcast i32 {{.+}} to float
; CHECK: fpext bfloat {{.+}} to float
; CHECK: call float @llvm.fma.f32
	v_fma_mix_f32_bf16 v10, v11, v12, v13 op_sel:[1,0,0] op_sel_hi:[1,0,1]
; CHECK: bitcast i32 {{.+}} to float
; CHECK: bitcast i32 {{.+}} to float
; CHECK: bitcast i32 {{.+}} to float
; CHECK: call float @llvm.fma.f32
	v_fma_mix_f32_bf16 v10, v11, v12, v13 op_sel:[1,1,1] op_sel_hi:[0,0,0]
; CHECK: call float @llvm.fabs.f32
; CHECK: fneg float
; CHECK: call float @llvm.fma.f32
; CHECK: call float @llvm.maxnum.f32
; CHECK: call float @llvm.minnum.f32
	v_fma_mix_f32_bf16 v10, -|v11|, v12, v13 op_sel_hi:[1,0,1] clamp
; CHECK: call float @llvm.fma.f32(float 1.000000e+00,
	v_fma_mix_f32_bf16 v10, 1.0, v11, 0 op_sel_hi:[1,0,0]
; CHECK: call float @llvm.fma.f32(float 1.000000e+00,
	v_fma_mix_f32_bf16 v10, 1.0, v11, 0 op_sel:[1,0,0] op_sel_hi:[1,0,0]
; CHECK: call float @llvm.fma.f32(float 1.000000e+00,
	v_fma_mix_f32_bf16 v10, 1.0, v11, 0 op_sel_hi:[0,0,0]
	s_mov_b32 s4, -1
; CHECK: [[COND:%.+]] = icmp ne i32 {{.+}}, 0
; CHECK: select i1 [[COND]], i32
	v_cndmask_b32_e64 v5, v6, v7, s4
; CHECK: ret void
	s_endpgm

	.globl	literal_f64
	.p2align	8
	.type	literal_f64,@function
; LITERAL-LABEL: define amdgpu_kernel void @literal_f64(
literal_f64:
; LITERAL: store i32 1065353216, ptr addrspace(1)
	v_cvt_f32_f64_e32 v2, lit(0x3ff00000)

	global_store_dword v[0:1], v2, off
; LITERAL: call i32 @llvm.fptosi.sat.i32.f64(double 1.000000e+00)
	v_cvt_i32_f64_e32 v2, lit(0x3ff00000)
; LITERAL: store i32 {{.+}}, ptr addrspace(1)
	global_store_dword v[0:1], v2, off
; LITERAL: call i32 @llvm.fptoui.sat.i32.f64(double 1.000000e+00)
	v_cvt_u32_f64_e32 v2, lit(0x3ff00000)
; LITERAL: store i32 {{.+}}, ptr addrspace(1)
	global_store_dword v[0:1], v2, off
; LITERAL: [[LIT64:%.+]] = phi i32 [ 1065353216, {{%.+}} ],
	v_cvt_f32_f64_e32 v2, lit64(0x3ff0000000000000)

; LITERAL: store i32 [[LIT64]], ptr addrspace(1)
	global_store_dword v[0:1], v2, off
; LITERAL: [[INLINE:%.+]] = phi i32 [ 1065353216, {{%.+}} ],
	v_cvt_f32_f64_e32 v2, 1.0

; LITERAL: store i32 [[INLINE]], ptr addrspace(1)
	global_store_dword v[0:1], v2, off
	s_endpgm

	.globl	refuse_clamp
	.p2align	8
	.type	refuse_clamp,@function
; REFUSE-CLAMP: unsupported-instruction-form: v_exp_f32 [VOP3]
; REFUSE-CLAMP-SAME: floating-point output clamp is not supported
refuse_clamp:
	v_exp_f32_e64 v0, v1 clamp
	s_endpgm

	.globl	refuse_omod
	.p2align	8
	.type	refuse_omod,@function
; REFUSE-OMOD: unsupported-instruction-form: v_exp_f32 [VOP3]
; REFUSE-OMOD-SAME: floating-point output multiplier is not supported
refuse_omod:
	v_exp_f32_e64 v0, v1 mul:2
	s_endpgm

	.globl	refuse_div_scale_modifiers
	.p2align	8
	.type	refuse_div_scale_modifiers,@function
; REFUSE-SCALE-MODS: unsupported-instruction-form: v_div_scale_f32 [VOP3]
; REFUSE-SCALE-MODS-SAME: asymmetric divide scale source modifiers
refuse_div_scale_modifiers:
	v_div_scale_f32 v0, vcc_lo, -v0, v1, v0
	s_endpgm

	.globl	true16_destination
	.p2align	8
	.type	true16_destination,@function
; TRUE16: fptrunc float
true16_destination:
	v_cvt_f16_f32_e32 v0.l, v1
	s_endpgm

	.globl	refuse_tanh
	.p2align	8
	.type	refuse_tanh,@function
; REFUSE-TANH: unsupported-instruction-form: v_tanh_f32 [VOP1]
; REFUSE-TANH-SAME: target does not support v_tanh_f32
; SUPPORT-TANH-LABEL: define amdgpu_kernel void @refuse_tanh(
; SUPPORT-TANH: call float @llvm.amdgcn.tanh.f32
; SUPPORT-TANH: store i32 {{.+}}, ptr addrspace(1)
refuse_tanh:
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, 0
	v_tanh_f32_e32 v0, v1
	global_store_dword v[2:3], v0, off
	s_endpgm

	.globl	refuse_bf16_denorm
	.p2align	8
	.type	refuse_bf16_denorm,@function
; REFUSE-BF16-DENORM: unsupported-floating-point-mode: v_fma_mix_f32_bf16
; REFUSE-BF16-DENORM-SAME: BF16 input denormal flushing is unsupported
refuse_bf16_denorm:
	v_fma_mix_f32_bf16 v0, v1, v2, v3 op_sel_hi:[1,0,0]
	s_endpgm

	.globl	refuse_fmac_clamp
	.p2align	8
	.type	refuse_fmac_clamp,@function
; REFUSE-FMAC-CLAMP: unsupported-instruction-form: v_fmac_f32 [VOP3]
; REFUSE-FMAC-CLAMP-SAME: floating-point output clamp is not supported
refuse_fmac_clamp:
	v_fmac_f32_e64 v0, v1, v2 clamp
	s_endpgm

	.globl	refuse_fmac_omod
	.p2align	8
	.type	refuse_fmac_omod,@function
; REFUSE-FMAC-OMOD: unsupported-instruction-form: v_fmac_f64 [VOP3]
; REFUSE-FMAC-OMOD-SAME: floating-point output multiplier is not supported
refuse_fmac_omod:
	v_fmac_f64_e64 v[0:1], v[2:3], v[4:5] mul:2
	s_endpgm

	.globl	refuse_f64_clamp
	.p2align	8
	.type	refuse_f64_clamp,@function
; REFUSE-F64-CLAMP: unsupported-instruction-form: v_rndne_f64 [VOP3]
; REFUSE-F64-CLAMP-SAME: floating-point output clamp is not supported
refuse_f64_clamp:
	v_rndne_f64_e64 v[0:1], v[2:3] clamp
	s_endpgm

	.globl	refuse_f64_omod
	.p2align	8
	.type	refuse_f64_omod,@function
; REFUSE-F64-OMOD: unsupported-instruction-form: v_ldexp_f64 [VOP3]
; REFUSE-F64-OMOD-SAME: floating-point output multiplier is not supported
refuse_f64_omod:
	v_ldexp_f64 v[0:1], v[2:3], v4 mul:2
	s_endpgm

	.globl	refuse_f64_rounding
	.p2align	8
	.type	refuse_f64_rounding,@function
; REFUSE-F64-ROUNDING: unsupported-floating-point-mode: v_trunc_f64 [VOP1]
; REFUSE-F64-ROUNDING-SAME: f64 rounding mode 1 is unsupported
refuse_f64_rounding:
	v_trunc_f64_e32 v[0:1], v[2:3]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vop_math
		.amdhsa_next_free_vgpr 51
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel vop3_math
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_next_free_vgpr 17
		.amdhsa_next_free_sgpr 5
	.end_amdhsa_kernel
	.amdhsa_kernel literal_f64
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_clamp
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_omod
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_div_scale_modifiers
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel true16_destination
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_tanh
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_bf16_denorm
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_fmac_clamp
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_fmac_omod
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_f64_clamp
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_f64_omod
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel refuse_f64_rounding
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_round_mode_16_64 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vop_math
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         vop_math.kd
    .vgpr_count:     51
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           vop3_math
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         vop3_math.kd
    .vgpr_count:     17
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           literal_f64
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         literal_f64.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_clamp
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_clamp.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_omod
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_omod.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_div_scale_modifiers
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_div_scale_modifiers.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           true16_destination
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         true16_destination.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_tanh
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_tanh.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_bf16_denorm
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_bf16_denorm.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_fmac_clamp
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_fmac_clamp.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_fmac_omod
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_fmac_omod.kd
    .vgpr_count:     6
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_f64_clamp
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_f64_clamp.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_f64_omod
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_f64_omod.kd
    .vgpr_count:     5
    .wavefront_size: 32
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_f64_rounding
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_f64_rounding.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
