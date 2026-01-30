//===--- BuiltinsAMDGPU.td - AMDGPU Builtin function defs -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGPU-specific builtin function database.
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

//===----------------------------------------------------------------------===//
// AMDGPU builtin base classes
//===----------------------------------------------------------------------===//

class AMDGPUBuiltin<string prototype, list<Attribute> Attr = [], string Feat = ""> : TargetBuiltin {
  let Spellings = [NAME];
  let Prototype = prototype;
  let Attributes = !listconcat([NoThrow], Attr);
  let Features = Feat;
}

//===----------------------------------------------------------------------===//
// SI+ only builtins.
//===----------------------------------------------------------------------===//

def __builtin_amdgcn_dispatch_ptr : AMDGPUBuiltin<"void address_space<4> *()", [Const]>;
def __builtin_amdgcn_kernarg_segment_ptr : AMDGPUBuiltin<"void address_space<4> *()", [Const]>;
def __builtin_amdgcn_implicitarg_ptr : AMDGPUBuiltin<"void address_space<4> *()", [Const]>;
def __builtin_amdgcn_queue_ptr : AMDGPUBuiltin<"void address_space<4> *()", [Const]>;

def __builtin_amdgcn_workgroup_id_x : AMDGPUBuiltin<"unsigned int()", [Const]>;
def __builtin_amdgcn_workgroup_id_y : AMDGPUBuiltin<"unsigned int()", [Const]>;
def __builtin_amdgcn_workgroup_id_z : AMDGPUBuiltin<"unsigned int()", [Const]>;

def __builtin_amdgcn_cluster_id_x : AMDGPUBuiltin<"unsigned int()", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cluster_id_y : AMDGPUBuiltin<"unsigned int()", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cluster_id_z : AMDGPUBuiltin<"unsigned int()", [Const], "gfx1250-insts">;

def __builtin_amdgcn_cluster_workgroup_id_x : AMDGPUBuiltin<"unsigned int()", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cluster_workgroup_id_y : AMDGPUBuiltin<"unsigned int()", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cluster_workgroup_id_z : AMDGPUBuiltin<"unsigned int()", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cluster_workgroup_flat_id : AMDGPUBuiltin<"unsigned int()", [Const], "gfx1250-insts">;

def __builtin_amdgcn_cluster_workgroup_max_id_x : AMDGPUBuiltin<"unsigned int()", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cluster_workgroup_max_id_y : AMDGPUBuiltin<"unsigned int()", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cluster_workgroup_max_id_z : AMDGPUBuiltin<"unsigned int()", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cluster_workgroup_max_flat_id : AMDGPUBuiltin<"unsigned int()", [Const], "gfx1250-insts">;

def __builtin_amdgcn_workitem_id_x : AMDGPUBuiltin<"unsigned int()", [Const]>;
def __builtin_amdgcn_workitem_id_y : AMDGPUBuiltin<"unsigned int()", [Const]>;
def __builtin_amdgcn_workitem_id_z : AMDGPUBuiltin<"unsigned int()", [Const]>;

def __builtin_amdgcn_workgroup_size_x : AMDGPUBuiltin<"unsigned short()", [Const]>;
def __builtin_amdgcn_workgroup_size_y : AMDGPUBuiltin<"unsigned short()", [Const]>;
def __builtin_amdgcn_workgroup_size_z : AMDGPUBuiltin<"unsigned short()", [Const]>;

def __builtin_amdgcn_grid_size_x : AMDGPUBuiltin<"unsigned int()", [Const]>;
def __builtin_amdgcn_grid_size_y : AMDGPUBuiltin<"unsigned int()", [Const]>;
def __builtin_amdgcn_grid_size_z : AMDGPUBuiltin<"unsigned int()", [Const]>;

def __builtin_amdgcn_mbcnt_hi : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int)", [Const]>;
def __builtin_amdgcn_mbcnt_lo : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int)", [Const]>;

def __builtin_amdgcn_s_memtime : AMDGPUBuiltin<"uint64_t()", [], "s-memtime-inst">;

//===----------------------------------------------------------------------===//
// Instruction builtins.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_s_getreg : AMDGPUBuiltin<"unsigned int(_Constant int)">;
def __builtin_amdgcn_s_setreg : AMDGPUBuiltin<"void(_Constant int, unsigned int)">;
def __builtin_amdgcn_s_getpc : AMDGPUBuiltin<"uint64_t()">;
def __builtin_amdgcn_s_waitcnt : AMDGPUBuiltin<"void(_Constant int)">;
def __builtin_amdgcn_s_sendmsg : AMDGPUBuiltin<"void(_Constant int, unsigned int)">;
def __builtin_amdgcn_s_sendmsghalt : AMDGPUBuiltin<"void(_Constant int, unsigned int)">;
def __builtin_amdgcn_s_barrier : AMDGPUBuiltin<"void()">;
def __builtin_amdgcn_s_ttracedata : AMDGPUBuiltin<"void(int)">;
def __builtin_amdgcn_wave_barrier : AMDGPUBuiltin<"void()">;
def __builtin_amdgcn_sched_barrier : AMDGPUBuiltin<"void(_Constant int)">;
def __builtin_amdgcn_sched_group_barrier : AMDGPUBuiltin<"void(_Constant int, _Constant int, _Constant int)">;
def __builtin_amdgcn_iglp_opt : AMDGPUBuiltin<"void(_Constant int)">;
def __builtin_amdgcn_s_dcache_inv : AMDGPUBuiltin<"void()">;
def __builtin_amdgcn_buffer_wbinvl1 : AMDGPUBuiltin<"void()">;
def __builtin_amdgcn_fence : AMDGPUBuiltin<"void(unsigned int, char const *, ...)">;
def __builtin_amdgcn_groupstaticsize : AMDGPUBuiltin<"unsigned int()">;
def __builtin_amdgcn_wavefrontsize : AMDGPUBuiltin<"unsigned int()", [Const]>;

def __builtin_amdgcn_atomic_inc32 : AMDGPUBuiltin<"uint32_t(uint32_t volatile *, uint32_t, unsigned int, char const *)">;
def __builtin_amdgcn_atomic_inc64 : AMDGPUBuiltin<"uint64_t(uint64_t volatile *, uint64_t, unsigned int, char const *)">;

def __builtin_amdgcn_atomic_dec32 : AMDGPUBuiltin<"uint32_t(uint32_t volatile *, uint32_t, unsigned int, char const *)">;
def __builtin_amdgcn_atomic_dec64 : AMDGPUBuiltin<"uint64_t(uint64_t volatile *, uint64_t, unsigned int, char const *)">;

// FIXME: Need to disallow constant address space.
def __builtin_amdgcn_div_scale : AMDGPUBuiltin<"double(double, double, bool, bool *)">;
def __builtin_amdgcn_div_scalef : AMDGPUBuiltin<"float(float, float, bool, bool *)">;
def __builtin_amdgcn_div_fmas : AMDGPUBuiltin<"double(double, double, double, bool)", [Const]>;
def __builtin_amdgcn_div_fmasf : AMDGPUBuiltin<"float(float, float, float, bool)", [Const]>;
def __builtin_amdgcn_div_fixup : AMDGPUBuiltin<"double(double, double, double)", [Const]>;
def __builtin_amdgcn_div_fixupf : AMDGPUBuiltin<"float(float, float, float)", [Const]>;
def __builtin_amdgcn_trig_preop : AMDGPUBuiltin<"double(double, int)", [Const]>;
def __builtin_amdgcn_trig_preopf : AMDGPUBuiltin<"float(float, int)", [Const]>;
def __builtin_amdgcn_rcp : AMDGPUBuiltin<"double(double)", [Const]>;
def __builtin_amdgcn_rcpf : AMDGPUBuiltin<"float(float)", [Const]>;
def __builtin_amdgcn_sqrt : AMDGPUBuiltin<"double(double)", [Const]>;
def __builtin_amdgcn_sqrtf : AMDGPUBuiltin<"float(float)", [Const]>;
def __builtin_amdgcn_rsq : AMDGPUBuiltin<"double(double)", [Const]>;
def __builtin_amdgcn_rsqf : AMDGPUBuiltin<"float(float)", [Const]>;
def __builtin_amdgcn_rsq_clamp : AMDGPUBuiltin<"double(double)", [Const]>;
def __builtin_amdgcn_rsq_clampf : AMDGPUBuiltin<"float(float)", [Const]>;
def __builtin_amdgcn_sinf : AMDGPUBuiltin<"float(float)", [Const]>;
def __builtin_amdgcn_cosf : AMDGPUBuiltin<"float(float)", [Const]>;
def __builtin_amdgcn_logf : AMDGPUBuiltin<"float(float)", [Const]>;
def __builtin_amdgcn_exp2f : AMDGPUBuiltin<"float(float)", [Const]>;
def __builtin_amdgcn_log_clampf : AMDGPUBuiltin<"float(float)", [Const]>;
def __builtin_amdgcn_ldexp : AMDGPUBuiltin<"double(double, int)", [Const]>;
def __builtin_amdgcn_ldexpf : AMDGPUBuiltin<"float(float, int)", [Const]>;
def __builtin_amdgcn_frexp_mant : AMDGPUBuiltin<"double(double)", [Const]>;
def __builtin_amdgcn_frexp_mantf : AMDGPUBuiltin<"float(float)", [Const]>;
def __builtin_amdgcn_frexp_exp : AMDGPUBuiltin<"int(double)", [Const]>;
def __builtin_amdgcn_frexp_expf : AMDGPUBuiltin<"int(float)", [Const]>;
def __builtin_amdgcn_fract : AMDGPUBuiltin<"double(double)", [Const]>;
def __builtin_amdgcn_fractf : AMDGPUBuiltin<"float(float)", [Const]>;
def __builtin_amdgcn_lerp : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", [Const], "lerp-inst">;
def __builtin_amdgcn_class : AMDGPUBuiltin<"bool(double, int)", [Const]>;
def __builtin_amdgcn_classf : AMDGPUBuiltin<"bool(float, int)", [Const]>;
def __builtin_amdgcn_cubeid : AMDGPUBuiltin<"float(float, float, float)", [Const], "cube-insts">;
def __builtin_amdgcn_cubesc : AMDGPUBuiltin<"float(float, float, float)", [Const], "cube-insts">;
def __builtin_amdgcn_cubetc : AMDGPUBuiltin<"float(float, float, float)", [Const], "cube-insts">;
def __builtin_amdgcn_cubema : AMDGPUBuiltin<"float(float, float, float)", [Const], "cube-insts">;
def __builtin_amdgcn_s_sleep : AMDGPUBuiltin<"void(_Constant int)">;
def __builtin_amdgcn_s_incperflevel : AMDGPUBuiltin<"void(_Constant int)">;
def __builtin_amdgcn_s_decperflevel : AMDGPUBuiltin<"void(_Constant int)">;
def __builtin_amdgcn_s_setprio : AMDGPUBuiltin<"void(_Constant short)">;
def __builtin_amdgcn_ds_swizzle : AMDGPUBuiltin<"int(int, _Constant int)", [Const]>;
def __builtin_amdgcn_ds_permute : AMDGPUBuiltin<"int(int, int)", [Const]>;
def __builtin_amdgcn_ds_bpermute : AMDGPUBuiltin<"int(int, int)", [Const]>;
def __builtin_amdgcn_readfirstlane : AMDGPUBuiltin<"int(int)", [Const]>;
def __builtin_amdgcn_readlane : AMDGPUBuiltin<"int(int, int)", [Const]>;
def __builtin_amdgcn_fmed3f : AMDGPUBuiltin<"float(float, float, float)", [Const]>;
def __builtin_amdgcn_ds_faddf : AMDGPUBuiltin<"float(float address_space<3> *, float, _Constant int, _Constant int, _Constant bool)">;
def __builtin_amdgcn_ds_fminf : AMDGPUBuiltin<"float(float address_space<3> *, float, _Constant int, _Constant int, _Constant bool)">;
def __builtin_amdgcn_ds_fmaxf : AMDGPUBuiltin<"float(float address_space<3> *, float, _Constant int, _Constant int, _Constant bool)">;
def __builtin_amdgcn_ds_append : AMDGPUBuiltin<"int(int address_space<3> *)">;
def __builtin_amdgcn_ds_consume : AMDGPUBuiltin<"int(int address_space<3> *)">;
def __builtin_amdgcn_alignbit : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", [Const]>;
def __builtin_amdgcn_alignbyte : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", [Const]>;
def __builtin_amdgcn_ubfe : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", [Const]>;
def __builtin_amdgcn_sbfe : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", [Const]>;
def __builtin_amdgcn_cvt_pkrtz : AMDGPUBuiltin<"_ExtVector<2, __fp16>(float, float)", [Const]>;
def __builtin_amdgcn_cvt_pknorm_i16 : AMDGPUBuiltin<"_ExtVector<2, short>(float, float)", [Const], "cvt-pknorm-vop2-insts">;
def __builtin_amdgcn_cvt_pknorm_u16 : AMDGPUBuiltin<"_ExtVector<2, unsigned short>(float, float)", [Const], "cvt-pknorm-vop2-insts">;
def __builtin_amdgcn_cvt_pk_i16 : AMDGPUBuiltin<"_ExtVector<2, short>(int, int)", [Const]>;
def __builtin_amdgcn_cvt_pk_u16 : AMDGPUBuiltin<"_ExtVector<2, unsigned short>(unsigned int, unsigned int)", [Const]>;
def __builtin_amdgcn_cvt_pk_u8_f32 : AMDGPUBuiltin<"unsigned int(float, unsigned int, unsigned int)", [Const]>;
def __builtin_amdgcn_cvt_off_f32_i4 : AMDGPUBuiltin<"float(int)", [Const]>;
def __builtin_amdgcn_msad_u8 : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", [Const]>;
def __builtin_amdgcn_sad_u8 : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", [Const], "sad-insts">;
def __builtin_amdgcn_sad_hi_u8 : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", [Const], "sad-insts">;
def __builtin_amdgcn_sad_u16 : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", [Const], "sad-insts">;
def __builtin_amdgcn_qsad_pk_u16_u8 : AMDGPUBuiltin<"uint64_t(uint64_t, unsigned int, uint64_t)", [Const], "qsad-insts">;
def __builtin_amdgcn_mqsad_pk_u16_u8 : AMDGPUBuiltin<"uint64_t(uint64_t, unsigned int, uint64_t)", [Const]>;
def __builtin_amdgcn_mqsad_u32_u8 : AMDGPUBuiltin<"_ExtVector<4, unsigned int>(uint64_t, unsigned int, _ExtVector<4, unsigned int>)", [Const]>;

def __builtin_amdgcn_make_buffer_rsrc : AMDGPUBuiltin<"__amdgpu_buffer_rsrc_t(void *, short, int64_t, int)", [Const]>;
def __builtin_amdgcn_raw_buffer_store_b8 : AMDGPUBuiltin<"void(unsigned char, __amdgpu_buffer_rsrc_t, int, int, _Constant int)">;
def __builtin_amdgcn_raw_buffer_store_b16 : AMDGPUBuiltin<"void(unsigned short, __amdgpu_buffer_rsrc_t, int, int, _Constant int)">;
def __builtin_amdgcn_raw_buffer_store_b32 : AMDGPUBuiltin<"void(unsigned int, __amdgpu_buffer_rsrc_t, int, int, _Constant int)">;
def __builtin_amdgcn_raw_buffer_store_b64 : AMDGPUBuiltin<"void(_ExtVector<2, unsigned int>, __amdgpu_buffer_rsrc_t, int, int, _Constant int)">;
def __builtin_amdgcn_raw_buffer_store_b96 : AMDGPUBuiltin<"void(_ExtVector<3, unsigned int>, __amdgpu_buffer_rsrc_t, int, int, _Constant int)">;
def __builtin_amdgcn_raw_buffer_store_b128 : AMDGPUBuiltin<"void(_ExtVector<4, unsigned int>, __amdgpu_buffer_rsrc_t, int, int, _Constant int)">;
def __builtin_amdgcn_raw_buffer_load_b8 : AMDGPUBuiltin<"unsigned char(__amdgpu_buffer_rsrc_t, int, int, _Constant int)">;
def __builtin_amdgcn_raw_buffer_load_b16 : AMDGPUBuiltin<"unsigned short(__amdgpu_buffer_rsrc_t, int, int, _Constant int)">;
def __builtin_amdgcn_raw_buffer_load_b32 : AMDGPUBuiltin<"unsigned int(__amdgpu_buffer_rsrc_t, int, int, _Constant int)">;
def __builtin_amdgcn_raw_buffer_load_b64 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(__amdgpu_buffer_rsrc_t, int, int, _Constant int)">;
def __builtin_amdgcn_raw_buffer_load_b96 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(__amdgpu_buffer_rsrc_t, int, int, _Constant int)">;
def __builtin_amdgcn_raw_buffer_load_b128 : AMDGPUBuiltin<"_ExtVector<4, unsigned int>(__amdgpu_buffer_rsrc_t, int, int, _Constant int)">;

def __builtin_amdgcn_raw_ptr_buffer_atomic_add_i32 : AMDGPUBuiltin<"int(int, __amdgpu_buffer_rsrc_t, int, int, _Constant int)">;

def __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32 : AMDGPUBuiltin<"float(float, __amdgpu_buffer_rsrc_t, int, int, _Constant int)", [], "atomic-fadd-rtn-insts">;
def __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16 : AMDGPUBuiltin<"_ExtVector<2, _Float16>(_ExtVector<2, _Float16>, __amdgpu_buffer_rsrc_t, int, int, _Constant int)", [], "atomic-buffer-global-pk-add-f16-insts">;

def __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32 : AMDGPUBuiltin<"float(float, __amdgpu_buffer_rsrc_t, int, int, _Constant int)", [], "atomic-fmin-fmax-global-f32">;
def __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32 : AMDGPUBuiltin<"float(float, __amdgpu_buffer_rsrc_t, int, int, _Constant int)", [], "atomic-fmin-fmax-global-f32">;
def __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64 : AMDGPUBuiltin<"double(double, __amdgpu_buffer_rsrc_t, int, int, _Constant int)", [], "atomic-fmin-fmax-global-f64">;
def __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64 : AMDGPUBuiltin<"double(double, __amdgpu_buffer_rsrc_t, int, int, _Constant int)", [], "atomic-fmin-fmax-global-f64">;

def __builtin_amdgcn_raw_ptr_buffer_load_lds : AMDGPUBuiltin<"void(__amdgpu_buffer_rsrc_t, void address_space<3> *, _Constant unsigned int, int, int, _Constant int, _Constant int)", [], "vmem-to-lds-load-insts">;
def __builtin_amdgcn_struct_ptr_buffer_load_lds : AMDGPUBuiltin<"void(__amdgpu_buffer_rsrc_t, void address_space<3> *, _Constant unsigned int, int, int, int, _Constant int, _Constant int)", [], "vmem-to-lds-load-insts">;

//===----------------------------------------------------------------------===//
// Ballot builtins.
//===----------------------------------------------------------------------===//

def __builtin_amdgcn_ballot_w32 : AMDGPUBuiltin<"uint32_t(bool)", [Const], "wavefrontsize32">;
def __builtin_amdgcn_ballot_w64 : AMDGPUBuiltin<"uint64_t(bool)", [Const]>;

def __builtin_amdgcn_inverse_ballot_w32 : AMDGPUBuiltin<"bool(uint32_t)", [Const], "wavefrontsize32">;
def __builtin_amdgcn_inverse_ballot_w64 : AMDGPUBuiltin<"bool(uint64_t)", [Const], "wavefrontsize64">;

// Deprecated intrinsics in favor of __builtin_amdgn_ballot_{w32|w64}
def __builtin_amdgcn_uicmp : AMDGPUBuiltin<"uint64_t(unsigned int, unsigned int, _Constant int)", [Const]>;
def __builtin_amdgcn_uicmpl : AMDGPUBuiltin<"uint64_t(uint64_t, uint64_t, _Constant int)", [Const]>;
def __builtin_amdgcn_sicmp : AMDGPUBuiltin<"uint64_t(int, int, _Constant int)", [Const]>;
def __builtin_amdgcn_sicmpl : AMDGPUBuiltin<"uint64_t(int64_t, int64_t, _Constant int)", [Const]>;
def __builtin_amdgcn_fcmp : AMDGPUBuiltin<"uint64_t(double, double, _Constant int)", [Const]>;
def __builtin_amdgcn_fcmpf : AMDGPUBuiltin<"uint64_t(float, float, _Constant int)", [Const]>;

//===----------------------------------------------------------------------===//
// Flat addressing builtins.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_is_shared : AMDGPUBuiltin<"bool(void const address_space<0> *)", [Const]>;
def __builtin_amdgcn_is_private : AMDGPUBuiltin<"bool(void const address_space<0> *)", [Const]>;

//===----------------------------------------------------------------------===//
// GWS builtins.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_ds_gws_init : AMDGPUBuiltin<"void(unsigned int, unsigned int)", [], "gws">;
def __builtin_amdgcn_ds_gws_barrier : AMDGPUBuiltin<"void(unsigned int, unsigned int)", [], "gws">;
def __builtin_amdgcn_ds_gws_sema_v : AMDGPUBuiltin<"void(unsigned int)", [], "gws">;
def __builtin_amdgcn_ds_gws_sema_br : AMDGPUBuiltin<"void(unsigned int, unsigned int)", [], "gws">;
def __builtin_amdgcn_ds_gws_sema_p : AMDGPUBuiltin<"void(unsigned int)", [], "gws">;

//===----------------------------------------------------------------------===//
// CI+ only builtins.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_s_dcache_inv_vol : AMDGPUBuiltin<"void()", [], "ci-insts">;
def __builtin_amdgcn_buffer_wbinvl1_vol : AMDGPUBuiltin<"void()", [], "ci-insts">;
def __builtin_amdgcn_ds_gws_sema_release_all : AMDGPUBuiltin<"void(unsigned int)", [], "ci-insts">;

//===----------------------------------------------------------------------===//
// Interpolation builtins.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_interp_p1_f16 : AMDGPUBuiltin<"float(float, unsigned int, unsigned int, bool, unsigned int)", [Const]>;
def __builtin_amdgcn_interp_p2_f16 : AMDGPUBuiltin<"__fp16(float, float, unsigned int, unsigned int, bool, unsigned int)", [Const]>;
def __builtin_amdgcn_interp_p1 : AMDGPUBuiltin<"float(float, unsigned int, unsigned int, unsigned int)", [Const]>;
def __builtin_amdgcn_interp_p2 : AMDGPUBuiltin<"float(float, float, unsigned int, unsigned int, unsigned int)", [Const]>;
def __builtin_amdgcn_interp_mov : AMDGPUBuiltin<"float(unsigned int, unsigned int, unsigned int, unsigned int)", [Const]>;

//===----------------------------------------------------------------------===//
// VI+ only builtins.
//===----------------------------------------------------------------------===//

def __builtin_amdgcn_div_fixuph : AMDGPUBuiltin<"__fp16(__fp16, __fp16, __fp16)", [Const], "16-bit-insts">;
def __builtin_amdgcn_rcph : AMDGPUBuiltin<"__fp16(__fp16)", [Const], "16-bit-insts">;
def __builtin_amdgcn_sqrth : AMDGPUBuiltin<"__fp16(__fp16)", [Const], "16-bit-insts">;
def __builtin_amdgcn_rsqh : AMDGPUBuiltin<"__fp16(__fp16)", [Const], "16-bit-insts">;
def __builtin_amdgcn_sinh : AMDGPUBuiltin<"__fp16(__fp16)", [Const], "16-bit-insts">;
def __builtin_amdgcn_cosh : AMDGPUBuiltin<"__fp16(__fp16)", [Const], "16-bit-insts">;
def __builtin_amdgcn_ldexph : AMDGPUBuiltin<"__fp16(__fp16, int)", [Const], "16-bit-insts">;
def __builtin_amdgcn_frexp_manth : AMDGPUBuiltin<"__fp16(__fp16)", [Const], "16-bit-insts">;
def __builtin_amdgcn_frexp_exph : AMDGPUBuiltin<"short(__fp16)", [Const], "16-bit-insts">;
def __builtin_amdgcn_fracth : AMDGPUBuiltin<"__fp16(__fp16)", [Const], "16-bit-insts">;
def __builtin_amdgcn_classh : AMDGPUBuiltin<"bool(__fp16, int)", [Const], "16-bit-insts">;
def __builtin_amdgcn_s_memrealtime : AMDGPUBuiltin<"uint64_t()", [], "s-memrealtime">;
def __builtin_amdgcn_mov_dpp : AMDGPUBuiltin<"int(int, _Constant int, _Constant int, _Constant int, _Constant bool)", [Const, CustomTypeChecking], "dpp">;
def __builtin_amdgcn_update_dpp : AMDGPUBuiltin<"int(int, int, _Constant int, _Constant int, _Constant int, _Constant bool)", [Const, CustomTypeChecking], "dpp">;
def __builtin_amdgcn_s_dcache_wb : AMDGPUBuiltin<"void()", [], "gfx8-insts">;
def __builtin_amdgcn_perm : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", [Const], "gfx8-insts">;

//===----------------------------------------------------------------------===//
// GFX9+ only builtins.
//===----------------------------------------------------------------------===//

def __builtin_amdgcn_fmed3h : AMDGPUBuiltin<"__fp16(__fp16, __fp16, __fp16)", [Const], "gfx9-insts">;

def __builtin_amdgcn_global_atomic_fadd_f64 : AMDGPUBuiltin<"double(double address_space<1> *, double)", [], "gfx90a-insts">;
def __builtin_amdgcn_global_atomic_fadd_f32 : AMDGPUBuiltin<"float(float address_space<1> *, float)", [], "atomic-fadd-rtn-insts">;
def __builtin_amdgcn_global_atomic_fadd_v2f16 : AMDGPUBuiltin<"_ExtVector<2, _Float16>(_ExtVector<2, _Float16 address_space<1> *>, _ExtVector<2, _Float16>)", [CustomTypeChecking], "atomic-buffer-global-pk-add-f16-insts">;
def __builtin_amdgcn_global_atomic_fmin_f64 : AMDGPUBuiltin<"double(double address_space<1> *, double)", [], "gfx90a-insts">;
def __builtin_amdgcn_global_atomic_fmax_f64 : AMDGPUBuiltin<"double(double address_space<1> *, double)", [], "gfx90a-insts">;

def __builtin_amdgcn_flat_atomic_fadd_f64 : AMDGPUBuiltin<"double(double address_space<0> *, double)", [], "gfx90a-insts">;
def __builtin_amdgcn_flat_atomic_fmin_f64 : AMDGPUBuiltin<"double(double address_space<0> *, double)", [], "gfx90a-insts">;
def __builtin_amdgcn_flat_atomic_fmax_f64 : AMDGPUBuiltin<"double(double address_space<0> *, double)", [], "gfx90a-insts">;

def __builtin_amdgcn_ds_atomic_fadd_f64 : AMDGPUBuiltin<"double(double address_space<3> *, double)", [], "gfx90a-insts">;
def __builtin_amdgcn_ds_atomic_fadd_f32 : AMDGPUBuiltin<"float(float address_space<3> *, float)", [], "gfx8-insts">;

def __builtin_amdgcn_flat_atomic_fadd_f32 : AMDGPUBuiltin<"float(float address_space<0> *, float)", [], "gfx940-insts">;
def __builtin_amdgcn_flat_atomic_fadd_v2f16 : AMDGPUBuiltin<"_ExtVector<2, _Float16>(_ExtVector<2, _Float16 address_space<0> *>, _ExtVector<2, _Float16>)", [CustomTypeChecking], "atomic-flat-pk-add-16-insts">;
def __builtin_amdgcn_flat_atomic_fadd_v2bf16 : AMDGPUBuiltin<"_ExtVector<2, short>(_ExtVector<2, short address_space<0> *>, _ExtVector<2, short>)", [CustomTypeChecking], "atomic-flat-pk-add-16-insts">;
def __builtin_amdgcn_global_atomic_fadd_v2bf16 : AMDGPUBuiltin<"_ExtVector<2, short>(_ExtVector<2, short address_space<1> *>, _ExtVector<2, short>)", [CustomTypeChecking], "atomic-global-pk-add-bf16-inst">;
def __builtin_amdgcn_ds_atomic_fadd_v2bf16 : AMDGPUBuiltin<"_ExtVector<2, short>(_ExtVector<2, short address_space<3> *>, _ExtVector<2, short>)", [CustomTypeChecking], "atomic-ds-pk-add-16-insts">;
def __builtin_amdgcn_ds_atomic_fadd_v2f16 : AMDGPUBuiltin<"_ExtVector<2, _Float16>(_ExtVector<2, _Float16 address_space<3> *>, _ExtVector<2, _Float16>)", [CustomTypeChecking], "atomic-ds-pk-add-16-insts">;
def __builtin_amdgcn_load_to_lds : AMDGPUBuiltin<"void(void *, void address_space<3> *, _Constant unsigned int, _Constant int, _Constant unsigned int)", [], "vmem-to-lds-load-insts">;
def __builtin_amdgcn_global_load_lds : AMDGPUBuiltin<"void(void address_space<1> *, void address_space<3> *, _Constant unsigned int, _Constant int, _Constant unsigned int)", [], "vmem-to-lds-load-insts">;

//===----------------------------------------------------------------------===//
// Deep learning builtins.
//===----------------------------------------------------------------------===//

def __builtin_amdgcn_fdot2 : AMDGPUBuiltin<"float(_ExtVector<2, _Float16>, _ExtVector<2, _Float16>, float, _Constant bool)", [Const], "dot10-insts">;
def __builtin_amdgcn_fdot2_f16_f16 : AMDGPUBuiltin<"_Float16(_ExtVector<2, _Float16>, _ExtVector<2, _Float16>, _Float16)", [Const], "dot9-insts">;
def __builtin_amdgcn_fdot2_bf16_bf16 : AMDGPUBuiltin<"short(_ExtVector<2, short>, _ExtVector<2, short>, short)", [Const], "dot9-insts">;
def __builtin_amdgcn_fdot2_f32_bf16 : AMDGPUBuiltin<"float(_ExtVector<2, short>, _ExtVector<2, short>, float, _Constant bool)", [Const], "dot12-insts">;
def __builtin_amdgcn_sdot2 : AMDGPUBuiltin<"int(_ExtVector<2, short>, _ExtVector<2, short>, int, _Constant bool)", [Const], "dot2-insts">;
def __builtin_amdgcn_udot2 : AMDGPUBuiltin<"unsigned int(_ExtVector<2, unsigned short>, _ExtVector<2, unsigned short>, unsigned int, _Constant bool)", [Const], "dot2-insts">;
def __builtin_amdgcn_sdot4 : AMDGPUBuiltin<"int(int, int, int, _Constant bool)", [Const], "dot1-insts">;
def __builtin_amdgcn_udot4 : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant bool)", [Const], "dot7-insts">;
def __builtin_amdgcn_sudot4 : AMDGPUBuiltin<"int(_Constant bool, int, _Constant bool, int, int, _Constant bool)", [Const], "dot8-insts">;
def __builtin_amdgcn_sdot8 : AMDGPUBuiltin<"int(int, int, int, _Constant bool)", [Const], "dot1-insts">;
def __builtin_amdgcn_udot8 : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant bool)", [Const], "dot7-insts">;
def __builtin_amdgcn_sudot8 : AMDGPUBuiltin<"int(_Constant bool, int, _Constant bool, int, int, _Constant bool)", [Const], "dot8-insts">;
def __builtin_amdgcn_dot4_f32_fp8_bf8 : AMDGPUBuiltin<"float(unsigned int, unsigned int, float)", [Const], "dot11-insts">;
def __builtin_amdgcn_dot4_f32_bf8_fp8 : AMDGPUBuiltin<"float(unsigned int, unsigned int, float)", [Const], "dot11-insts">;
def __builtin_amdgcn_dot4_f32_fp8_fp8 : AMDGPUBuiltin<"float(unsigned int, unsigned int, float)", [Const], "dot11-insts">;
def __builtin_amdgcn_dot4_f32_bf8_bf8 : AMDGPUBuiltin<"float(unsigned int, unsigned int, float)", [Const], "dot11-insts">;
def __builtin_amdgcn_fdot2c_f32_bf16 : AMDGPUBuiltin<"float(_ExtVector<2, __bf16>, _ExtVector<2, __bf16>, float, _Constant bool)", [Const], "dot13-insts">;

//===----------------------------------------------------------------------===//
// GFX10+ only builtins.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_permlane16 : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, unsigned int, _Constant bool, _Constant bool)", [Const], "gfx10-insts">;
def __builtin_amdgcn_permlanex16 : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, unsigned int, _Constant bool, _Constant bool)", [Const], "gfx10-insts">;
def __builtin_amdgcn_mov_dpp8 : AMDGPUBuiltin<"unsigned int(unsigned int, _Constant unsigned int)", [Const, CustomTypeChecking], "gfx10-insts">;
def __builtin_amdgcn_s_ttracedata_imm : AMDGPUBuiltin<"void(_Constant short)", [], "gfx10-insts">;

//===----------------------------------------------------------------------===//
// Raytracing builtins.
// By default the 1st argument is i32 and the 4/5-th arguments are float4.
// Postfix l indicates the 1st argument is i64.
// Postfix h indicates the 4/5-th arguments are half4.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_image_bvh_intersect_ray : AMDGPUBuiltin<"_ExtVector<4, unsigned int>(unsigned int, float, _ExtVector<4, float>, _ExtVector<4, float>, _ExtVector<4, float>, _ExtVector<4, unsigned int>)", [Const], "gfx10-insts">;
def __builtin_amdgcn_image_bvh_intersect_ray_h : AMDGPUBuiltin<"_ExtVector<4, unsigned int>(unsigned int, float, _ExtVector<4, float>, _ExtVector<4, _Float16>, _ExtVector<4, _Float16>, _ExtVector<4, unsigned int>)", [Const], "gfx10-insts">;
def __builtin_amdgcn_image_bvh_intersect_ray_l : AMDGPUBuiltin<"_ExtVector<4, unsigned int>(uint64_t, float, _ExtVector<4, float>, _ExtVector<4, float>, _ExtVector<4, float>, _ExtVector<4, unsigned int>)", [Const], "gfx10-insts">;
def __builtin_amdgcn_image_bvh_intersect_ray_lh : AMDGPUBuiltin<"_ExtVector<4, unsigned int>(uint64_t, float, _ExtVector<4, float>, _ExtVector<4, _Float16>, _ExtVector<4, _Float16>, _ExtVector<4, unsigned int>)", [Const], "gfx10-insts">;


//===----------------------------------------------------------------------===//
// GFX11+ only builtins.
//===----------------------------------------------------------------------===//

// TODO: This is a no-op in wave32. Should the builtin require wavefrontsize64?
def __builtin_amdgcn_permlane64 : AMDGPUBuiltin<"unsigned int(unsigned int)", [Const], "gfx11-insts">;
def __builtin_amdgcn_s_wait_event_export_ready : AMDGPUBuiltin<"void()", [], "gfx11-insts">;

//===----------------------------------------------------------------------===//
// WMMA builtins.
// Postfix w32 indicates the builtin requires wavefront size of 32.
// Postfix w64 indicates the builtin requires wavefront size of 64.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<16, _Float16>, _ExtVector<16, _Float16>, _ExtVector<8, float>)", [Const], "gfx11-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<16, short>, _ExtVector<16, short>, _ExtVector<8, float>)", [Const], "gfx11-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x16_f16_w32 : AMDGPUBuiltin<"_ExtVector<16, _Float16>(_ExtVector<16, _Float16>, _ExtVector<16, _Float16>, _ExtVector<16, _Float16>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32 : AMDGPUBuiltin<"_ExtVector<16, short>(_ExtVector<16, short>, _ExtVector<16, short>, _ExtVector<16, short>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32 : AMDGPUBuiltin<"_ExtVector<16, _Float16>(_ExtVector<16, _Float16>, _ExtVector<16, _Float16>, _ExtVector<16, _Float16>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w32 : AMDGPUBuiltin<"_ExtVector<16, short>(_ExtVector<16, short>, _ExtVector<16, short>, _ExtVector<16, short>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 : AMDGPUBuiltin<"_ExtVector<8, int>(_Constant bool, _ExtVector<4, int>, _Constant bool, _ExtVector<4, int>, _ExtVector<8, int>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32 : AMDGPUBuiltin<"_ExtVector<8, int>(_Constant bool, _ExtVector<2, int>, _Constant bool, _ExtVector<2, int>, _ExtVector<8, int>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize32">;

def __builtin_amdgcn_wmma_f32_16x16x16_f16_w64 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<16, _Float16>, _ExtVector<16, _Float16>, _ExtVector<4, float>)", [Const], "gfx11-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<16, short>, _ExtVector<16, short>, _ExtVector<4, float>)", [Const], "gfx11-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_f16_16x16x16_f16_w64 : AMDGPUBuiltin<"_ExtVector<8, _Float16>(_ExtVector<16, _Float16>, _ExtVector<16, _Float16>, _ExtVector<8, _Float16>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64 : AMDGPUBuiltin<"_ExtVector<8, short>(_ExtVector<16, short>, _ExtVector<16, short>, _ExtVector<8, short>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w64 : AMDGPUBuiltin<"_ExtVector<8, _Float16>(_ExtVector<16, _Float16>, _ExtVector<16, _Float16>, _ExtVector<8, _Float16>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w64 : AMDGPUBuiltin<"_ExtVector<8, short>(_ExtVector<16, short>, _ExtVector<16, short>, _ExtVector<8, short>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64 : AMDGPUBuiltin<"_ExtVector<4, int>(_Constant bool, _ExtVector<4, int>, _Constant bool, _ExtVector<4, int>, _ExtVector<4, int>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64 : AMDGPUBuiltin<"_ExtVector<4, int>(_Constant bool, _ExtVector<2, int>, _Constant bool, _ExtVector<2, int>, _ExtVector<4, int>, _Constant bool)", [Const], "gfx11-insts,wavefrontsize64">;

def __builtin_amdgcn_s_sendmsg_rtn : AMDGPUBuiltin<"unsigned int(_Constant unsigned int)", [], "gfx11-insts">;
def __builtin_amdgcn_s_sendmsg_rtnl : AMDGPUBuiltin<"uint64_t(_Constant unsigned int)", [], "gfx11-insts">;

def __builtin_amdgcn_ds_bvh_stack_rtn : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(unsigned int, unsigned int, _ExtVector<4, unsigned int>, _Constant int)", [], "gfx11-insts">;

//===----------------------------------------------------------------------===//
// Special builtins.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_read_exec : AMDGPUBuiltin<"uint64_t()", [Const]>;
def __builtin_amdgcn_read_exec_lo : AMDGPUBuiltin<"unsigned int()", [Const]>;
def __builtin_amdgcn_read_exec_hi : AMDGPUBuiltin<"unsigned int()", [Const]>;

def __builtin_amdgcn_endpgm : AMDGPUBuiltin<"void()", [NoReturn]>;

def __builtin_amdgcn_get_fpenv : AMDGPUBuiltin<"uint64_t()">;
def __builtin_amdgcn_set_fpenv : AMDGPUBuiltin<"void(uint64_t)">;

//===----------------------------------------------------------------------===//

// Wave Reduction builtins.

//===----------------------------------------------------------------------===//

def __builtin_amdgcn_wave_reduce_add_u32 : AMDGPUBuiltin<"uint32_t(uint32_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_sub_u32 : AMDGPUBuiltin<"uint32_t(uint32_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_min_i32 : AMDGPUBuiltin<"int32_t(int32_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_min_u32 : AMDGPUBuiltin<"uint32_t(uint32_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_max_i32 : AMDGPUBuiltin<"int32_t(int32_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_max_u32 : AMDGPUBuiltin<"uint32_t(uint32_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_and_b32 : AMDGPUBuiltin<"int32_t(int32_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_or_b32 : AMDGPUBuiltin<"int32_t(int32_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_xor_b32 : AMDGPUBuiltin<"int32_t(int32_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_add_u64 : AMDGPUBuiltin<"uint64_t(uint64_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_sub_u64 : AMDGPUBuiltin<"uint64_t(uint64_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_min_i64 : AMDGPUBuiltin<"int64_t(int64_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_min_u64 : AMDGPUBuiltin<"uint64_t(uint64_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_max_i64 : AMDGPUBuiltin<"int64_t(int64_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_max_u64 : AMDGPUBuiltin<"uint64_t(uint64_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_and_b64 : AMDGPUBuiltin<"int64_t(int64_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_or_b64 : AMDGPUBuiltin<"int64_t(int64_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_xor_b64 : AMDGPUBuiltin<"int64_t(int64_t, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_fadd_f32 : AMDGPUBuiltin<"float(float, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_fsub_f32 : AMDGPUBuiltin<"float(float, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_fmin_f32 : AMDGPUBuiltin<"float(float, int32_t)", [Const]>;
def __builtin_amdgcn_wave_reduce_fmax_f32 : AMDGPUBuiltin<"float(float, int32_t)", [Const]>;

//===----------------------------------------------------------------------===//
// R600-NI only builtins.
//===----------------------------------------------------------------------===//

def __builtin_r600_implicitarg_ptr : AMDGPUBuiltin<"unsigned char address_space<7> *()", [Const]>;

def __builtin_r600_read_tgid_x : AMDGPUBuiltin<"unsigned int()", [Const]>;
def __builtin_r600_read_tgid_y : AMDGPUBuiltin<"unsigned int()", [Const]>;
def __builtin_r600_read_tgid_z : AMDGPUBuiltin<"unsigned int()", [Const]>;

def __builtin_r600_read_tidig_x : AMDGPUBuiltin<"unsigned int()", [Const]>;
def __builtin_r600_read_tidig_y : AMDGPUBuiltin<"unsigned int()", [Const]>;
def __builtin_r600_read_tidig_z : AMDGPUBuiltin<"unsigned int()", [Const]>;

def __builtin_r600_recipsqrt_ieee : AMDGPUBuiltin<"double(double)", [Const]>;
def __builtin_r600_recipsqrt_ieeef : AMDGPUBuiltin<"float(float)", [Const]>;

//===----------------------------------------------------------------------===//
// MFMA builtins.
//===----------------------------------------------------------------------===//

def __builtin_amdgcn_mfma_f32_32x32x1f32 : AMDGPUBuiltin<"_ExtVector<32, float>(float, float, _ExtVector<32, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_16x16x1f32 : AMDGPUBuiltin<"_ExtVector<16, float>(float, float, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_4x4x1f32 : AMDGPUBuiltin<"_ExtVector<4, float>(float, float, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_32x32x2f32 : AMDGPUBuiltin<"_ExtVector<16, float>(float, float, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_16x16x4f32 : AMDGPUBuiltin<"_ExtVector<4, float>(float, float, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_32x32x4f16 : AMDGPUBuiltin<"_ExtVector<32, float>(_ExtVector<4, _Float16>, _ExtVector<4, _Float16>, _ExtVector<32, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_16x16x4f16 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<4, _Float16>, _ExtVector<4, _Float16>, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_4x4x4f16 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, _Float16>, _ExtVector<4, _Float16>, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_32x32x8f16 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<4, _Float16>, _ExtVector<4, _Float16>, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_16x16x16f16 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, _Float16>, _ExtVector<4, _Float16>, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_i32_32x32x4i8 : AMDGPUBuiltin<"_ExtVector<32, int>(int, int, _ExtVector<32, int>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_i32_16x16x4i8 : AMDGPUBuiltin<"_ExtVector<16, int>(int, int, _ExtVector<16, int>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_i32_4x4x4i8 : AMDGPUBuiltin<"_ExtVector<4, int>(int, int, _ExtVector<4, int>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_i32_32x32x8i8 : AMDGPUBuiltin<"_ExtVector<16, int>(int, int, _ExtVector<16, int>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_i32_16x16x16i8 : AMDGPUBuiltin<"_ExtVector<4, int>(int, int, _ExtVector<4, int>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_32x32x2bf16 : AMDGPUBuiltin<"_ExtVector<32, float>(_ExtVector<2, short>, _ExtVector<2, short>, _ExtVector<32, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_16x16x2bf16 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<2, short>, _ExtVector<2, short>, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_4x4x2bf16 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<2, short>, _ExtVector<2, short>, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_32x32x4bf16 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<2, short>, _ExtVector<2, short>, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_16x16x8bf16 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<2, short>, _ExtVector<2, short>, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;

def __builtin_amdgcn_mfma_f32_32x32x4bf16_1k : AMDGPUBuiltin<"_ExtVector<32, float>(_ExtVector<4, short>, _ExtVector<4, short>, _ExtVector<32, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_16x16x4bf16_1k : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<4, short>, _ExtVector<4, short>, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_4x4x4bf16_1k : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, short>, _ExtVector<4, short>, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_32x32x8bf16_1k : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<4, short>, _ExtVector<4, short>, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_16x16x16bf16_1k : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, short>, _ExtVector<4, short>, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f64_16x16x4f64 : AMDGPUBuiltin<"_ExtVector<4, double>(double, double, _ExtVector<4, double>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f64_4x4x4f64 : AMDGPUBuiltin<"double(double, double, double, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;

def __builtin_amdgcn_mfma_i32_16x16x32_i8 : AMDGPUBuiltin<"_ExtVector<4, int>(int64_t, int64_t, _ExtVector<4, int>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_i32_32x32x16_i8 : AMDGPUBuiltin<"_ExtVector<16, int>(int64_t, int64_t, _ExtVector<16, int>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_16x16x8_xf32 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<2, float>, _ExtVector<2, float>, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_32x32x4_xf32 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<2, float>, _ExtVector<2, float>, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<4, float>(int64_t, int64_t, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<4, float>(int64_t, int64_t, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<4, float>(int64_t, int64_t, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<4, float>(int64_t, int64_t, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<16, float>(int64_t, int64_t, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<16, float>(int64_t, int64_t, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<16, float>(int64_t, int64_t, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<16, float>(int64_t, int64_t, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_smfmac_f32_16x16x32_f16 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, _Float16>, _ExtVector<8, _Float16>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x16_f16 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<4, _Float16>, _ExtVector<8, _Float16>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_smfmac_f32_16x16x32_bf16 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, short>, _ExtVector<8, short>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x16_bf16 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<4, short>, _ExtVector<8, short>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_smfmac_i32_16x16x64_i8 : AMDGPUBuiltin<"_ExtVector<4, int>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<4, int>, int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_smfmac_i32_32x32x32_i8 : AMDGPUBuiltin<"_ExtVector<16, int>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<16, int>, int, _Constant int, _Constant int)", [Const], "mai-insts">;
def __builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "fp8-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "fp8-insts">;

def __builtin_amdgcn_cvt_f32_bf8 : AMDGPUBuiltin<"float(int, _Constant int)", [Const], "fp8-conversion-insts">;
def __builtin_amdgcn_cvt_f32_fp8 : AMDGPUBuiltin<"float(int, _Constant int)", [Const], "fp8-conversion-insts">;
def __builtin_amdgcn_cvt_f32_fp8_e5m3 : AMDGPUBuiltin<"float(int, _Constant int)", [Const], "fp8e5m3-insts">;
def __builtin_amdgcn_cvt_pk_f32_bf8 : AMDGPUBuiltin<"_ExtVector<2, float>(int, _Constant bool)", [Const], "fp8-conversion-insts">;
def __builtin_amdgcn_cvt_pk_f32_fp8 : AMDGPUBuiltin<"_ExtVector<2, float>(int, _Constant bool)", [Const], "fp8-conversion-insts">;
def __builtin_amdgcn_cvt_pk_bf8_f32 : AMDGPUBuiltin<"int(float, float, int, _Constant bool)", [Const], "fp8-conversion-insts">;
def __builtin_amdgcn_cvt_pk_fp8_f32 : AMDGPUBuiltin<"int(float, float, int, _Constant bool)", [Const], "fp8-conversion-insts">;
def __builtin_amdgcn_cvt_sr_bf8_f32 : AMDGPUBuiltin<"int(float, int, int, _Constant int)", [Const], "fp8-conversion-insts">;
def __builtin_amdgcn_cvt_sr_fp8_f32 : AMDGPUBuiltin<"int(float, int, int, _Constant int)", [Const], "fp8-conversion-insts">;

//===----------------------------------------------------------------------===//
// GFX950 only builtins.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<8, int32_t>, _ExtVector<8, int32_t>, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int, int, _Constant int, int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<8, int32_t>, _ExtVector<8, int32_t>, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int, int, _Constant int, int)", [Const], "gfx950-insts">;

def __builtin_amdgcn_mfma_f32_16x16x32_f16 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<8, _Float16>, _ExtVector<8, _Float16>, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_mfma_f32_16x16x32_bf16 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<8, __bf16>, _ExtVector<8, __bf16>, _ExtVector<4, float>, _Constant int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_mfma_f32_32x32x16_f16 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<8, _Float16>, _ExtVector<8, _Float16>, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_mfma_f32_32x32x16_bf16 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<8, __bf16>, _ExtVector<8, __bf16>, _ExtVector<16, float>, _Constant int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_mfma_i32_16x16x64_i8 : AMDGPUBuiltin<"_ExtVector<4, int>(_ExtVector<4, int>, _ExtVector<4, int>, _ExtVector<4, int>, _Constant int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_mfma_i32_32x32x32_i8 : AMDGPUBuiltin<"_ExtVector<16, int>(_ExtVector<4, int>, _ExtVector<4, int>, _ExtVector<16, int>, _Constant int, _Constant int, _Constant int)", [Const], "gfx950-insts">;

def __builtin_amdgcn_smfmac_f32_16x16x64_f16 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<8, _Float16>, _ExtVector<16, _Float16>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x32_f16 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<8, _Float16>, _ExtVector<16, _Float16>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_f32_16x16x64_bf16 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<8, __bf16>, _ExtVector<16, __bf16>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x32_bf16 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<8, __bf16>, _ExtVector<16, __bf16>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_i32_16x16x128_i8 : AMDGPUBuiltin<"_ExtVector<4, int>(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<4, int>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_i32_32x32x64_i8 : AMDGPUBuiltin<"_ExtVector<16, int>(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<16, int>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_f32_16x16x128_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_f32_16x16x128_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_f32_16x16x128_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<4, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x64_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x64_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x64_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;
def __builtin_amdgcn_smfmac_f32_32x32x64_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<16, float>, int, _Constant int, _Constant int)", [Const], "gfx950-insts">;

def __builtin_amdgcn_permlane16_swap : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(unsigned int, unsigned int, _Constant bool, _Constant bool)", [Const], "permlane16-swap">;
def __builtin_amdgcn_permlane32_swap : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(unsigned int, unsigned int, _Constant bool, _Constant bool)", [Const], "permlane32-swap">;

def __builtin_amdgcn_ds_read_tr4_b64_v2i32 : AMDGPUBuiltin<"_ExtVector<2, int>(_ExtVector<2, int address_space<3> *>)", [Const], "gfx950-insts">;
def __builtin_amdgcn_ds_read_tr6_b96_v3i32 : AMDGPUBuiltin<"_ExtVector<3, int>(_ExtVector<3, int address_space<3> *>)", [Const], "gfx950-insts">;
def __builtin_amdgcn_ds_read_tr8_b64_v2i32 : AMDGPUBuiltin<"_ExtVector<2, int>(_ExtVector<2, int address_space<3> *>)", [Const], "gfx950-insts">;
def __builtin_amdgcn_ds_read_tr16_b64_v4i16 : AMDGPUBuiltin<"_ExtVector<4, short>(_ExtVector<4, short address_space<3> *>)", [Const], "gfx950-insts">;
def __builtin_amdgcn_ds_read_tr16_b64_v4f16 : AMDGPUBuiltin<"_ExtVector<4, __fp16>(_ExtVector<4, __fp16 address_space<3> *>)", [Const], "gfx950-insts">;
def __builtin_amdgcn_ds_read_tr16_b64_v4bf16 : AMDGPUBuiltin<"_ExtVector<4, __bf16>(_ExtVector<4, __bf16 address_space<3> *>)", [Const], "gfx950-insts">;

def __builtin_amdgcn_ashr_pk_i8_i32 : AMDGPUBuiltin<"unsigned short(unsigned int, unsigned int, unsigned int)", [Const], "ashr-pk-insts">;
def __builtin_amdgcn_ashr_pk_u8_i32 : AMDGPUBuiltin<"unsigned short(unsigned int, unsigned int, unsigned int)", [Const], "ashr-pk-insts">;

def __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<16, float>, _ExtVector<16, float>, float)", [Const], "gfx950-insts">;
def __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<16, float>, _ExtVector<16, float>, float)", [Const], "gfx950-insts">;

//===----------------------------------------------------------------------===//
// GFX12+ only builtins.
//===----------------------------------------------------------------------===//

def __builtin_amdgcn_s_sleep_var : AMDGPUBuiltin<"void(unsigned int)", [], "gfx12-insts">;
def __builtin_amdgcn_permlane16_var : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant bool, _Constant bool)", [Const], "gfx12-insts">;
def __builtin_amdgcn_permlanex16_var : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant bool, _Constant bool)", [Const], "gfx12-insts">;
def __builtin_amdgcn_s_barrier_signal : AMDGPUBuiltin<"void(_Constant int)", [], "gfx12-insts">;
def __builtin_amdgcn_s_barrier_signal_var : AMDGPUBuiltin<"void(void *, int)", [], "gfx12-insts">;
def __builtin_amdgcn_s_barrier_wait : AMDGPUBuiltin<"void(_Constant short)", [], "gfx12-insts">;
def __builtin_amdgcn_s_barrier_signal_isfirst : AMDGPUBuiltin<"bool(_Constant int)", [], "gfx12-insts">;
def __builtin_amdgcn_s_barrier_init : AMDGPUBuiltin<"void(void *, int)", [], "gfx12-insts">;
def __builtin_amdgcn_s_barrier_join : AMDGPUBuiltin<"void(void *)", [], "gfx12-insts">;
def __builtin_amdgcn_s_barrier_leave : AMDGPUBuiltin<"void(_Constant short)", [], "gfx12-insts">;
def __builtin_amdgcn_s_get_barrier_state : AMDGPUBuiltin<"unsigned int(int)", [], "gfx12-insts">;
def __builtin_amdgcn_s_get_named_barrier_state : AMDGPUBuiltin<"unsigned int(void *)", [], "gfx12-insts">;
def __builtin_amdgcn_s_prefetch_data : AMDGPUBuiltin<"void(void const *, unsigned int)", [Const], "gfx12-insts">;
def __builtin_amdgcn_s_buffer_prefetch_data : AMDGPUBuiltin<"void(__amdgpu_buffer_rsrc_t, _Constant int, unsigned int)", [Const], "gfx12-insts">;

def __builtin_amdgcn_global_load_tr_b64_v2i32 : AMDGPUBuiltin<"_ExtVector<2, int>(_ExtVector<2, int address_space<1> *>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_global_load_tr_b128_v8i16 : AMDGPUBuiltin<"_ExtVector<8, short>(_ExtVector<8, short address_space<1> *>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_global_load_tr_b128_v8f16 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, __fp16 address_space<1> *>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_global_load_tr_b128_v8bf16 : AMDGPUBuiltin<"_ExtVector<8, __bf16>(_ExtVector<8, __bf16 address_space<1> *>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_global_load_tr_b64_i32 : AMDGPUBuiltin<"int(int address_space<1> *)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_global_load_tr_b128_v4i16 : AMDGPUBuiltin<"_ExtVector<4, short>(_ExtVector<4, short address_space<1> *>)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_global_load_tr_b128_v4f16 : AMDGPUBuiltin<"_ExtVector<4, __fp16>(_ExtVector<4, __fp16 address_space<1> *>)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_global_load_tr_b128_v4bf16 : AMDGPUBuiltin<"_ExtVector<4, __bf16>(_ExtVector<4, __bf16 address_space<1> *>)", [Const], "gfx12-insts,wavefrontsize64">;

def __builtin_amdgcn_ds_bpermute_fi_b32 : AMDGPUBuiltin<"int(int, int)", [Const], "gfx12-insts">;

// For the following two builtins, the second and third return values of the
// intrinsics are returned through the last two pointer-type function arguments.
def __builtin_amdgcn_image_bvh8_intersect_ray : AMDGPUBuiltin<"_ExtVector<10, unsigned int>(uint64_t, float, unsigned char, _ExtVector<3, float>, _ExtVector<3, float>, unsigned int, _ExtVector<4, unsigned int>, _ExtVector<3, float *>, _ExtVector<3, float *>)", [Const], "gfx12-insts">;
def __builtin_amdgcn_image_bvh_dual_intersect_ray : AMDGPUBuiltin<"_ExtVector<10, unsigned int>(uint64_t, float, unsigned char, _ExtVector<3, float>, _ExtVector<3, float>, _ExtVector<2, unsigned int>, _ExtVector<4, unsigned int>, _ExtVector<3, float *>, _ExtVector<3, float *>)", [Const], "gfx12-insts">;

def __builtin_amdgcn_ds_bvh_stack_push4_pop1_rtn : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(unsigned int, unsigned int, _ExtVector<4, unsigned int>, _Constant int)", [], "gfx11-insts">;
def __builtin_amdgcn_ds_bvh_stack_push8_pop1_rtn : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(unsigned int, unsigned int, _ExtVector<8, unsigned int>, _Constant int)", [], "gfx12-insts">;

// The intrinsic returns {i64, i32}, the builtin returns <2 x i64>.
// The second return value of the intrinsic is zext'ed.
def __builtin_amdgcn_ds_bvh_stack_push8_pop2_rtn : AMDGPUBuiltin<"_ExtVector<2, uint64_t>(unsigned int, unsigned int, _ExtVector<8, unsigned int>, _Constant int)", [], "gfx12-insts">;

//===----------------------------------------------------------------------===//
// WMMA builtins.
// Postfix w32 indicates the builtin requires wavefront size of 32.
// Postfix w64 indicates the builtin requires wavefront size of 64.
//
// Some of these are very similar to their GFX11 counterparts, but they don't
// require replication of the A,B matrices, so they use fewer vector elements.
// Therefore, we add an "_gfx12" suffix to distinguish them from the existing
// builtins.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, _Float16>, _ExtVector<8, _Float16>, _ExtVector<8, float>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, short>, _ExtVector<8, short>, _ExtVector<8, float>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12 : AMDGPUBuiltin<"_ExtVector<8, _Float16>(_ExtVector<8, _Float16>, _ExtVector<8, _Float16>, _ExtVector<8, _Float16>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12 : AMDGPUBuiltin<"_ExtVector<8, short>(_ExtVector<8, short>, _ExtVector<8, short>, _ExtVector<8, short>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12 : AMDGPUBuiltin<"_ExtVector<8, int>(_Constant bool, _ExtVector<2, int>, _Constant bool, _ExtVector<2, int>, _ExtVector<8, int>, _Constant bool)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12 : AMDGPUBuiltin<"_ExtVector<8, int>(_Constant bool, int, _Constant bool, int, _ExtVector<8, int>, _Constant bool)", [Const], "gfx12-insts,wavefrontsize32">;
// These are gfx12-only, but for consistency with the other WMMA variants we're
// keeping the "_gfx12" suffix.
def __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<2, int>, _ExtVector<2, int>, _ExtVector<8, float>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<2, int>, _ExtVector<2, int>, _ExtVector<8, float>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<2, int>, _ExtVector<2, int>, _ExtVector<8, float>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<2, int>, _ExtVector<2, int>, _ExtVector<8, float>)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12 : AMDGPUBuiltin<"_ExtVector<8, int>(_Constant bool, _ExtVector<2, int>, _Constant bool, _ExtVector<2, int>, _ExtVector<8, int>, _Constant bool)", [Const], "gfx12-insts,wavefrontsize32">;

def __builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, _Float16>, _ExtVector<4, _Float16>, _ExtVector<4, float>)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, short>, _ExtVector<4, short>, _ExtVector<4, float>)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_f16_16x16x16_f16_w64_gfx12 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(_ExtVector<4, _Float16>, _ExtVector<4, _Float16>, _ExtVector<4, _Float16>)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64_gfx12 : AMDGPUBuiltin<"_ExtVector<4, short>(_ExtVector<4, short>, _ExtVector<4, short>, _ExtVector<4, short>)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12 : AMDGPUBuiltin<"_ExtVector<4, int>(_Constant bool, int, _Constant bool, int, _ExtVector<4, int>, _Constant bool)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64_gfx12 : AMDGPUBuiltin<"_ExtVector<4, int>(_Constant bool, int, _Constant bool, int, _ExtVector<4, int>, _Constant bool)", [Const], "gfx12-insts,wavefrontsize64">;
// These are gfx12-only, but for consistency with the other WMMA variants we're
// keeping the "_gfx12" suffix.
def __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w64_gfx12 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, _ExtVector<4, float>)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w64_gfx12 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, _ExtVector<4, float>)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w64_gfx12 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, _ExtVector<4, float>)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w64_gfx12 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, _ExtVector<4, float>)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12 : AMDGPUBuiltin<"_ExtVector<4, int>(_Constant bool, int, _Constant bool, int, _ExtVector<4, int>, _Constant bool)", [Const], "gfx12-insts,wavefrontsize64">;

def __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, __fp16>, _ExtVector<16, __fp16>, _ExtVector<8, float>, int)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, short>, _ExtVector<16, short>, _ExtVector<8, float>, int)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f16_16x16x32_f16_w32 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, __fp16>, _ExtVector<16, __fp16>, _ExtVector<8, __fp16>, int)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32 : AMDGPUBuiltin<"_ExtVector<8, short>(_ExtVector<8, short>, _ExtVector<16, short>, _ExtVector<8, short>, int)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32 : AMDGPUBuiltin<"_ExtVector<8, int>(_Constant bool, _ExtVector<2, int>, _Constant bool, _ExtVector<4, int>, _ExtVector<8, int>, int, _Constant bool)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32 : AMDGPUBuiltin<"_ExtVector<8, int>(_Constant bool, int, _Constant bool, _ExtVector<2, int>, _ExtVector<8, int>, int, _Constant bool)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32 : AMDGPUBuiltin<"_ExtVector<8, int>(_Constant bool, _ExtVector<2, int>, _Constant bool, _ExtVector<4, int>, _ExtVector<8, int>, int, _Constant bool)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<8, float>, int)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<8, float>, int)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<8, float>, int)", [Const], "gfx12-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<2, int>, _ExtVector<4, int>, _ExtVector<8, float>, int)", [Const], "gfx12-insts,wavefrontsize32">;

def __builtin_amdgcn_swmmac_f32_16x16x32_f16_w64 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, __fp16>, _ExtVector<8, __fp16>, _ExtVector<4, float>, int)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w64 : AMDGPUBuiltin<"_ExtVector<4, float>(_ExtVector<4, short>, _ExtVector<8, short>, _ExtVector<4, float>, int)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_swmmac_f16_16x16x32_f16_w64 : AMDGPUBuiltin<"_ExtVector<4, __fp16>(_ExtVector<4, __fp16>, _ExtVector<8, __fp16>, _ExtVector<4, __fp16>, int)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w64 : AMDGPUBuiltin<"_ExtVector<4, short>(_ExtVector<4, short>, _ExtVector<8, short>, _ExtVector<4, short>, int)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w64 : AMDGPUBuiltin<"_ExtVector<4, int>(_Constant bool, int, _Constant bool, _ExtVector<2, int>, _ExtVector<4, int>, int, _Constant bool)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w64 : AMDGPUBuiltin<"_ExtVector<4, int>(_Constant bool, int, _Constant bool, int, _ExtVector<4, int>, int, _Constant bool)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w64 : AMDGPUBuiltin<"_ExtVector<4, int>(_Constant bool, int, _Constant bool, _ExtVector<2, int>, _ExtVector<4, int>, int, _Constant bool)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w64 : AMDGPUBuiltin<"_ExtVector<4, float>(int, _ExtVector<2, int>, _ExtVector<4, float>, int)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w64 : AMDGPUBuiltin<"_ExtVector<4, float>(int, _ExtVector<2, int>, _ExtVector<4, float>, int)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w64 : AMDGPUBuiltin<"_ExtVector<4, float>(int, _ExtVector<2, int>, _ExtVector<4, float>, int)", [Const], "gfx12-insts,wavefrontsize64">;
def __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64 : AMDGPUBuiltin<"_ExtVector<4, float>(int, _ExtVector<2, int>, _ExtVector<4, float>, int)", [Const], "gfx12-insts,wavefrontsize64">;

def __builtin_amdgcn_prng_b32 : AMDGPUBuiltin<"unsigned int(unsigned int)", [Const], "prng-inst">;
def __builtin_amdgcn_cvt_scalef32_pk32_fp6_f16 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<32, _Float16>, float)", [Const], "f16bf16-to-fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk32_bf6_f16 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<32, _Float16>, float)", [Const], "f16bf16-to-fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<32, __bf16>, float)", [Const], "f16bf16-to-fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<32, __bf16>, float)", [Const], "f16bf16-to-fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_f16_fp8 : AMDGPUBuiltin<"_ExtVector<2, _Float16>(_ExtVector<2, _Float16>, int, float, _Constant int, _Constant bool)", [Const], "fp8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_f16_bf8 : AMDGPUBuiltin<"_ExtVector<2, _Float16>(_ExtVector<2, _Float16>, int, float, _Constant int, _Constant bool)", [Const], "bf8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_f32_fp8 : AMDGPUBuiltin<"float(int, float, _Constant int)", [Const], "fp8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_f32_bf8 : AMDGPUBuiltin<"float(int, float, _Constant int)", [Const], "bf8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_fp8_f32 : AMDGPUBuiltin<"_ExtVector<2, short>(_ExtVector<2, short>, float, float, float, _Constant bool)", [Const], "fp8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_bf8_f32 : AMDGPUBuiltin<"_ExtVector<2, short>(_ExtVector<2, short>, float, float, float, _Constant bool)", [Const], "bf8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_f32_fp8 : AMDGPUBuiltin<"_ExtVector<2, float>(unsigned int, float, _Constant bool)", [Const], "fp8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_f32_bf8 : AMDGPUBuiltin<"_ExtVector<2, float>(unsigned int, float, _Constant bool)", [Const], "bf8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_fp8_f16 : AMDGPUBuiltin<"_ExtVector<2, short>(_ExtVector<2, short>, _ExtVector<2, _Float16>, float, _Constant bool)", [Const], "fp8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16 : AMDGPUBuiltin<"_ExtVector<2, short>(_ExtVector<2, short>, _ExtVector<2, __bf16>, float, _Constant bool)", [Const], "fp8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_bf8_f16 : AMDGPUBuiltin<"_ExtVector<2, short>(_ExtVector<2, short>, _ExtVector<2, _Float16>, float, _Constant bool)", [Const], "bf8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16 : AMDGPUBuiltin<"_ExtVector<2, short>(_ExtVector<2, short>, _ExtVector<2, __bf16>, float, _Constant bool)", [Const], "bf8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_f32_fp4 : AMDGPUBuiltin<"_ExtVector<2, float>(unsigned int, float, _Constant int)", [Const], "fp4-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_fp4_f32 : AMDGPUBuiltin<"unsigned int(unsigned int, float, float, float, _Constant int)", [Const], "fp4-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_f16_fp4 : AMDGPUBuiltin<"_ExtVector<2, _Float16>(unsigned int, float, _Constant int)", [Const], "fp4-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4 : AMDGPUBuiltin<"_ExtVector<2, __bf16>(unsigned int, float, _Constant int)", [Const], "fp4-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6 : AMDGPUBuiltin<"_ExtVector<32, float>(_ExtVector<6, unsigned int>, float)", [Const], "fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6 : AMDGPUBuiltin<"_ExtVector<32, float>(_ExtVector<6, unsigned int>, float)", [Const], "fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk32_f16_fp6 : AMDGPUBuiltin<"_ExtVector<32, _Float16>(_ExtVector<6, unsigned int>, float)", [Const], "fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6 : AMDGPUBuiltin<"_ExtVector<32, __bf16>(_ExtVector<6, unsigned int>, float)", [Const], "fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk32_f16_bf6 : AMDGPUBuiltin<"_ExtVector<32, _Float16>(_ExtVector<6, unsigned int>, float)", [Const], "fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6 : AMDGPUBuiltin<"_ExtVector<32, __bf16>(_ExtVector<6, unsigned int>, float)", [Const], "fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_f16_fp8 : AMDGPUBuiltin<"_ExtVector<2, _Float16>(unsigned int, float, _Constant bool)", [Const], "fp8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8 : AMDGPUBuiltin<"_ExtVector<2, __bf16>(unsigned int, float, _Constant bool)", [Const], "fp8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_f16_bf8 : AMDGPUBuiltin<"_ExtVector<2, _Float16>(unsigned int, float, _Constant bool)", [Const], "bf8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8 : AMDGPUBuiltin<"_ExtVector<2, __bf16>(unsigned int, float, _Constant bool)", [Const], "bf8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_fp4_f16 : AMDGPUBuiltin<"unsigned int(unsigned int, _ExtVector<2, _Float16>, float, _Constant int)", [Const], "fp4-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16 : AMDGPUBuiltin<"unsigned int(unsigned int, _ExtVector<2, __bf16>, float, _Constant int)", [Const], "fp4-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16 : AMDGPUBuiltin<"unsigned int(unsigned int, _ExtVector<2, _Float16>, unsigned int, float, _Constant int)", [Const], "fp4-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16 : AMDGPUBuiltin<"unsigned int(unsigned int, _ExtVector<2, __bf16>, unsigned int, float, _Constant int)", [Const], "fp4-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32 : AMDGPUBuiltin<"unsigned int(unsigned int, _ExtVector<2, float>, unsigned int, float, _Constant int)", [Const], "fp4-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_bf8_bf16 : AMDGPUBuiltin<"int(int, __bf16, unsigned int, float, _Constant int)", [Const], "bf8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_bf8_f16 : AMDGPUBuiltin<"int(int, _Float16, unsigned int, float, _Constant int)", [Const], "bf8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_bf8_f32 : AMDGPUBuiltin<"int(int, float, unsigned int, float, _Constant int)", [Const], "bf8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_fp8_bf16 : AMDGPUBuiltin<"int(int, __bf16, unsigned int, float, _Constant int)", [Const], "fp8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_fp8_f16 : AMDGPUBuiltin<"int(int, _Float16, unsigned int, float, _Constant int)", [Const], "fp8-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_fp8_f32 : AMDGPUBuiltin<"int(int, float, unsigned int, float, _Constant int)", [Const], "fp8-cvt-scale-insts">;

def __builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_bf16 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<32, __bf16>, unsigned int, float)", [Const], "f16bf16-to-fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f16 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<32, _Float16>, unsigned int, float)", [Const], "f16bf16-to-fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk32_bf6_f32 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<32, float>, unsigned int, float)", [Const], "f16bf16-to-fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_bf16 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<32, __bf16>, unsigned int, float)", [Const], "f16bf16-to-fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f16 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<32, _Float16>, unsigned int, float)", [Const], "f16bf16-to-fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32 : AMDGPUBuiltin<"_ExtVector<6, unsigned int>(_ExtVector<32, float>, unsigned int, float)", [Const], "f16bf16-to-fp6bf6-cvt-scale-insts">;
def __builtin_amdgcn_bitop3_b32 : AMDGPUBuiltin<"int(int, int, int, _Constant unsigned int)", [Const], "bitop3-insts">;
def __builtin_amdgcn_bitop3_b16 : AMDGPUBuiltin<"short(short, short, short, _Constant unsigned int)", [Const], "bitop3-insts">;

def __builtin_amdgcn_cvt_sr_bf16_f32 : AMDGPUBuiltin<"_ExtVector<2, __bf16>(_ExtVector<2, __bf16>, float, unsigned int, _Constant bool)", [Const], "f32-to-f16bf16-cvt-sr-insts">;
def __builtin_amdgcn_cvt_sr_f16_f32 : AMDGPUBuiltin<"_ExtVector<2, _Float16>(_ExtVector<2, _Float16>, float, unsigned int, _Constant bool)", [Const], "f32-to-f16bf16-cvt-sr-insts">;

//===----------------------------------------------------------------------===//
// GFX1250+ only builtins.
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_s_cluster_barrier : AMDGPUBuiltin<"void()", [], "gfx1250-insts">;

def __builtin_amdgcn_flat_prefetch : AMDGPUBuiltin<"void(void const address_space<0> *, _Constant int)", [Const], "vmem-pref-insts">;
def __builtin_amdgcn_global_prefetch : AMDGPUBuiltin<"void(void const address_space<1> *, _Constant int)", [Const], "vmem-pref-insts">;

def __builtin_amdgcn_global_load_monitor_b32 : AMDGPUBuiltin<"int(int address_space<1> *, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_global_load_monitor_b64 : AMDGPUBuiltin<"_ExtVector<2, int>(_ExtVector<2, int address_space<1> *>, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_global_load_monitor_b128 : AMDGPUBuiltin<"_ExtVector<4, int>(_ExtVector<4, int address_space<1> *>, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_flat_load_monitor_b32 : AMDGPUBuiltin<"int(int address_space<0> *, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_flat_load_monitor_b64 : AMDGPUBuiltin<"_ExtVector<2, int>(_ExtVector<2, int address_space<0> *>, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_flat_load_monitor_b128 : AMDGPUBuiltin<"_ExtVector<4, int>(_ExtVector<4, int address_space<0> *>, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cluster_load_b32 : AMDGPUBuiltin<"int(int address_space<1> *, _Constant int, int)", [Const], "mcast-load-insts,wavefrontsize32">;
def __builtin_amdgcn_cluster_load_b64 : AMDGPUBuiltin<"_ExtVector<2, int>(_ExtVector<2, int address_space<1> *>, _Constant int, int)", [Const], "mcast-load-insts,wavefrontsize32">;
def __builtin_amdgcn_cluster_load_b128 : AMDGPUBuiltin<"_ExtVector<4, int>(_ExtVector<4, int address_space<1> *>, _Constant int, int)", [Const], "mcast-load-insts,wavefrontsize32">;
def __builtin_amdgcn_cluster_load_async_to_lds_b8 : AMDGPUBuiltin<"void(char address_space<1> *, char address_space<3> *, _Constant int, _Constant int, int)", [Const], "mcast-load-insts,wavefrontsize32">;
def __builtin_amdgcn_cluster_load_async_to_lds_b32 : AMDGPUBuiltin<"void(int address_space<1> *, int address_space<3> *, _Constant int, _Constant int, int)", [Const], "mcast-load-insts,wavefrontsize32">;
def __builtin_amdgcn_cluster_load_async_to_lds_b64 : AMDGPUBuiltin<"void(_ExtVector<2, int address_space<1> *>, _ExtVector<2, int address_space<3> *>, _Constant int, _Constant int, int)", [Const], "mcast-load-insts,wavefrontsize32">;
def __builtin_amdgcn_cluster_load_async_to_lds_b128 : AMDGPUBuiltin<"void(_ExtVector<4, int address_space<1> *>, _ExtVector<4, int address_space<3> *>, _Constant int, _Constant int, int)", [Const], "mcast-load-insts,wavefrontsize32">;
def __builtin_amdgcn_global_load_async_to_lds_b8 : AMDGPUBuiltin<"void(char address_space<1> *, char address_space<3> *, _Constant int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_global_load_async_to_lds_b32 : AMDGPUBuiltin<"void(int address_space<1> *, int address_space<3> *, _Constant int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_global_load_async_to_lds_b64 : AMDGPUBuiltin<"void(_ExtVector<2, int address_space<1> *>, _ExtVector<2, int address_space<3> *>, _Constant int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_global_load_async_to_lds_b128 : AMDGPUBuiltin<"void(_ExtVector<4, int address_space<1> *>, _ExtVector<4, int address_space<3> *>, _Constant int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_global_store_async_from_lds_b8 : AMDGPUBuiltin<"void(char address_space<1> *, char address_space<3> *, _Constant int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_global_store_async_from_lds_b32 : AMDGPUBuiltin<"void(int address_space<1> *, int address_space<3> *, _Constant int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_global_store_async_from_lds_b64 : AMDGPUBuiltin<"void(_ExtVector<2, int address_space<1> *>, _ExtVector<2, int address_space<3> *>, _Constant int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_global_store_async_from_lds_b128 : AMDGPUBuiltin<"void(_ExtVector<4, int address_space<1> *>, _ExtVector<4, int address_space<3> *>, _Constant int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_ds_atomic_async_barrier_arrive_b64 : AMDGPUBuiltin<"void(long int address_space<3> *)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_ds_atomic_barrier_arrive_rtn_b64 : AMDGPUBuiltin<"long int(long int address_space<3> *, long int)", [Const], "gfx1250-insts">;

def __builtin_amdgcn_tensor_load_to_lds : AMDGPUBuiltin<"void(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<4, int>, _ExtVector<4, int>, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_tensor_load_to_lds_d2 : AMDGPUBuiltin<"void(_ExtVector<4, int>, _ExtVector<8, int>, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_tensor_store_from_lds : AMDGPUBuiltin<"void(_ExtVector<4, int>, _ExtVector<8, int>, _ExtVector<4, int>, _ExtVector<4, int>, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_tensor_store_from_lds_d2 : AMDGPUBuiltin<"void(_ExtVector<4, int>, _ExtVector<8, int>, _Constant int)", [Const], "gfx1250-insts">;

def __builtin_amdgcn_global_load_tr4_b64_v2i32 : AMDGPUBuiltin<"_ExtVector<2, int>(_ExtVector<2, int address_space<1> *>)", [Const], "transpose-load-f4f6-insts,wavefrontsize32">;
def __builtin_amdgcn_global_load_tr8_b64_v2i32 : AMDGPUBuiltin<"_ExtVector<2, int>(_ExtVector<2, int address_space<1> *>)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_global_load_tr6_b96_v3i32 : AMDGPUBuiltin<"_ExtVector<3, int>(_ExtVector<3, int address_space<1> *>)", [Const], "transpose-load-f4f6-insts,wavefrontsize32">;
def __builtin_amdgcn_global_load_tr16_b128_v8i16 : AMDGPUBuiltin<"_ExtVector<8, short>(_ExtVector<8, short address_space<1> *>)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_global_load_tr16_b128_v8f16 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, __fp16 address_space<1> *>)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_global_load_tr16_b128_v8bf16 : AMDGPUBuiltin<"_ExtVector<8, __bf16>(_ExtVector<8, __bf16 address_space<1> *>)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_ds_load_tr4_b64_v2i32 : AMDGPUBuiltin<"_ExtVector<2, int>(_ExtVector<2, int address_space<3> *>)", [Const], "transpose-load-f4f6-insts,wavefrontsize32">;
def __builtin_amdgcn_ds_load_tr8_b64_v2i32 : AMDGPUBuiltin<"_ExtVector<2, int>(_ExtVector<2, int address_space<3> *>)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_ds_load_tr6_b96_v3i32 : AMDGPUBuiltin<"_ExtVector<3, int>(_ExtVector<3, int address_space<3> *>)", [Const], "transpose-load-f4f6-insts,wavefrontsize32">;
def __builtin_amdgcn_ds_load_tr16_b128_v8i16 : AMDGPUBuiltin<"_ExtVector<8, short>(_ExtVector<8, short address_space<3> *>)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_ds_load_tr16_b128_v8f16 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, __fp16 address_space<3> *>)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_ds_load_tr16_b128_v8bf16 : AMDGPUBuiltin<"_ExtVector<8, __bf16>(_ExtVector<8, __bf16 address_space<3> *>)", [Const], "gfx1250-insts,wavefrontsize32">;

def __builtin_amdgcn_s_setprio_inc_wg : AMDGPUBuiltin<"void(_Constant short)", [], "setprio-inc-wg-inst">;
def __builtin_amdgcn_s_monitor_sleep : AMDGPUBuiltin<"void(_Constant short)", [], "gfx1250-insts">;
def __builtin_amdgcn_s_wakeup_barrier : AMDGPUBuiltin<"void(void *)", [], "s-wakeup-barrier-inst">;

def __builtin_amdgcn_s_wait_asynccnt : AMDGPUBuiltin<"void(_Constant unsigned short)", [], "gfx1250-insts">;
def __builtin_amdgcn_s_wait_tensorcnt : AMDGPUBuiltin<"void(_Constant unsigned short)", [], "gfx1250-insts">;

def __builtin_amdgcn_tanhf : AMDGPUBuiltin<"float(float)", [Const], "tanh-insts">;
def __builtin_amdgcn_tanhh : AMDGPUBuiltin<"__fp16(__fp16)", [Const], "tanh-insts">;
def __builtin_amdgcn_tanh_bf16 : AMDGPUBuiltin<"__bf16(__bf16)", [Const], "bf16-trans-insts">;
def __builtin_amdgcn_rcp_bf16 : AMDGPUBuiltin<"__bf16(__bf16)", [Const], "bf16-trans-insts">;
def __builtin_amdgcn_sqrt_bf16 : AMDGPUBuiltin<"__bf16(__bf16)", [Const], "bf16-trans-insts">;
def __builtin_amdgcn_rsq_bf16 : AMDGPUBuiltin<"__bf16(__bf16)", [Const], "bf16-trans-insts">;
def __builtin_amdgcn_log_bf16 : AMDGPUBuiltin<"__bf16(__bf16)", [Const], "bf16-trans-insts">;
def __builtin_amdgcn_exp2_bf16 : AMDGPUBuiltin<"__bf16(__bf16)", [Const], "bf16-trans-insts">;
def __builtin_amdgcn_sin_bf16 : AMDGPUBuiltin<"__bf16(__bf16)", [Const], "bf16-trans-insts">;
def __builtin_amdgcn_cos_bf16 : AMDGPUBuiltin<"__bf16(__bf16)", [Const], "bf16-trans-insts">;

def __builtin_amdgcn_cvt_sr_pk_bf16_f32 : AMDGPUBuiltin<"_ExtVector<2, __bf16>(float, float, int)", [Const], "bf16-cvt-insts">;
def __builtin_amdgcn_cvt_sr_pk_f16_f32 : AMDGPUBuiltin<"_ExtVector<2, __fp16>(float, float, int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_f16_fp8 : AMDGPUBuiltin<"__fp16(int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_f16_bf8 : AMDGPUBuiltin<"__fp16(int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_pk_f16_fp8 : AMDGPUBuiltin<"_ExtVector<2, __fp16>(short)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_pk_f16_bf8 : AMDGPUBuiltin<"_ExtVector<2, __fp16>(short)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_pk_fp8_f16 : AMDGPUBuiltin<"short(_ExtVector<2, __fp16>)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_pk_bf8_f16 : AMDGPUBuiltin<"short(_ExtVector<2, __fp16>)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_sr_fp8_f16 : AMDGPUBuiltin<"int(__fp16, int, unsigned int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_sr_bf8_f16 : AMDGPUBuiltin<"int(__fp16, int, unsigned int, _Constant int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk8_f16_fp8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<2, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk8_bf16_fp8 : AMDGPUBuiltin<"_ExtVector<8, __bf16>(_ExtVector<2, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk8_f16_bf8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<2, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk8_bf16_bf8 : AMDGPUBuiltin<"_ExtVector<8, __bf16>(_ExtVector<2, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk8_f16_fp4 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(unsigned int, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk8_bf16_fp4 : AMDGPUBuiltin<"_ExtVector<8, __bf16>(unsigned int, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk8_f32_fp8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<2, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk8_f32_bf8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<2, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk8_f32_fp4 : AMDGPUBuiltin<"_ExtVector<8, float>(unsigned int, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk16_f16_fp6 : AMDGPUBuiltin<"_ExtVector<16, __fp16>(_ExtVector<3, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk16_bf16_fp6 : AMDGPUBuiltin<"_ExtVector<16, __bf16>(_ExtVector<3, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk16_f16_bf6 : AMDGPUBuiltin<"_ExtVector<16, __fp16>(_ExtVector<3, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk16_bf16_bf6 : AMDGPUBuiltin<"_ExtVector<16, __bf16>(_ExtVector<3, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk16_f32_fp6 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<3, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scale_pk16_f32_bf6 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<3, unsigned int>, unsigned int, _Constant unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, __bf16>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk8_bf8_bf16 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, __bf16>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk8_fp8_f16 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, __fp16>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk8_bf8_f16 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, __fp16>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, float>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk8_bf8_f32 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, float>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32 : AMDGPUBuiltin<"unsigned int(_ExtVector<8, float>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk8_fp4_f16 : AMDGPUBuiltin<"unsigned int(_ExtVector<8, __fp16>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16 : AMDGPUBuiltin<"unsigned int(_ExtVector<8, __bf16>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk16_fp6_f32 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, float>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk16_bf6_f32 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, float>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk16_fp6_f16 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, __fp16>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk16_bf6_f16 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, __fp16>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk16_fp6_bf16 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, __bf16>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_pk16_bf6_bf16 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, __bf16>, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_bf16 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, __bf16>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_bf16 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, __bf16>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f16 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, __fp16>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f16 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, __fp16>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk8_fp8_f32 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, float>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk8_bf8_f32 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<8, float>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_f32 : AMDGPUBuiltin<"unsigned int(_ExtVector<8, float>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_f16 : AMDGPUBuiltin<"unsigned int(_ExtVector<8, __fp16>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk8_fp4_bf16 : AMDGPUBuiltin<"unsigned int(_ExtVector<8, __bf16>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_bf16 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, __bf16>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_f16 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, __fp16>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk16_bf6_f32 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, float>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_bf16 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, __bf16>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_f16 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, __fp16>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_scalef32_sr_pk16_fp6_f32 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(_ExtVector<16, float>, unsigned int, float)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_cvt_pk_fp8_f32_e5m3 : AMDGPUBuiltin<"int(float, float, int, _Constant bool)", [Const], "fp8e5m3-insts">;
def __builtin_amdgcn_cvt_sr_fp8_f32_e5m3 : AMDGPUBuiltin<"int(float, int, int, _Constant int)", [Const], "fp8e5m3-insts">;
def __builtin_amdgcn_sat_pk4_i4_i8 : AMDGPUBuiltin<"unsigned short(unsigned int)", [Const], "gfx1250-insts">;
def __builtin_amdgcn_sat_pk4_u4_u8 : AMDGPUBuiltin<"unsigned short(unsigned int)", [Const], "gfx1250-insts">;

def __builtin_amdgcn_permlane_bcast : AMDGPUBuiltin<"int(int, int, int)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_permlane_up : AMDGPUBuiltin<"int(int, int, int)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_permlane_down : AMDGPUBuiltin<"int(int, int, int)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_permlane_xor : AMDGPUBuiltin<"int(int, int, int)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_permlane_idx_gen : AMDGPUBuiltin<"int(int, int)", [Const], "gfx1250-insts,wavefrontsize32">;

def __builtin_amdgcn_perm_pk16_b4_u4 : AMDGPUBuiltin<"_ExtVector<2, unsigned int>(unsigned int, unsigned int, _ExtVector<2, unsigned int>)", [Const], "tensor-cvt-lut-insts">;
def __builtin_amdgcn_perm_pk16_b6_u4 : AMDGPUBuiltin<"_ExtVector<3, unsigned int>(unsigned int, unsigned long int, _ExtVector<2, unsigned int>)", [Const], "tensor-cvt-lut-insts">;
def __builtin_amdgcn_perm_pk16_b8_u4 : AMDGPUBuiltin<"_ExtVector<4, unsigned int>(unsigned long int, unsigned long int, _ExtVector<2, unsigned int>)", [Const], "tensor-cvt-lut-insts">;

def __builtin_amdgcn_add_max_i32 : AMDGPUBuiltin<"int(int, int, int, _Constant bool)", [Const], "add-min-max-insts">;
def __builtin_amdgcn_add_max_u32 : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant bool)", [Const], "add-min-max-insts">;
def __builtin_amdgcn_add_min_i32 : AMDGPUBuiltin<"int(int, int, int, _Constant bool)", [Const], "add-min-max-insts">;
def __builtin_amdgcn_add_min_u32 : AMDGPUBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant bool)", [Const], "add-min-max-insts">;
def __builtin_amdgcn_pk_add_max_i16 : AMDGPUBuiltin<"_ExtVector<2, short>(_ExtVector<2, short>, _ExtVector<2, short>, _ExtVector<2, short>, _Constant bool)", [Const], "pk-add-min-max-insts">;
def __builtin_amdgcn_pk_add_max_u16 : AMDGPUBuiltin<"_ExtVector<2, unsigned short>(_ExtVector<2, unsigned short>, _ExtVector<2, unsigned short>, _ExtVector<2, unsigned short>, _Constant bool)", [Const], "pk-add-min-max-insts">;
def __builtin_amdgcn_pk_add_min_i16 : AMDGPUBuiltin<"_ExtVector<2, short>(_ExtVector<2, short>, _ExtVector<2, short>, _ExtVector<2, short>, _Constant bool)", [Const], "pk-add-min-max-insts">;
def __builtin_amdgcn_pk_add_min_u16 : AMDGPUBuiltin<"_ExtVector<2, unsigned short>(_ExtVector<2, unsigned short>, _ExtVector<2, unsigned short>, _ExtVector<2, unsigned short>, _Constant bool)", [Const], "pk-add-min-max-insts">;

// GFX1250 WMMA builtins
def __builtin_amdgcn_wmma_f32_16x16x4_f32 : AMDGPUBuiltin<"_ExtVector<8, float>(_Constant bool, _ExtVector<2, float>, _Constant bool, _ExtVector<2, float>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x32_bf16 : AMDGPUBuiltin<"_ExtVector<8, float>(_Constant bool, _ExtVector<16, __bf16>, _Constant bool, _ExtVector<16, __bf16>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_bf16_16x16x32_bf16 : AMDGPUBuiltin<"_ExtVector<8, __bf16>(_Constant bool, _ExtVector<16, __bf16>, _Constant bool, _ExtVector<16, __bf16>, _Constant short, _ExtVector<8, __bf16>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_bf16f32_16x16x32_bf16 : AMDGPUBuiltin<"_ExtVector<8, __bf16>(_Constant bool, _ExtVector<16, __bf16>, _Constant bool, _ExtVector<16, __bf16>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, int>, _ExtVector<8, int>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, int>, _ExtVector<8, int>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, int>, _ExtVector<8, int>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, int>, _ExtVector<8, int>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, int>, _ExtVector<8, int>, _Constant short, _ExtVector<8, __fp16>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, int>, _ExtVector<8, int>, _Constant short, _ExtVector<8, __fp16>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, int>, _ExtVector<8, int>, _Constant short, _ExtVector<8, __fp16>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, int>, _ExtVector<8, int>, _Constant short, _ExtVector<8, __fp16>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_i32_16x16x64_iu8 : AMDGPUBuiltin<"_ExtVector<8, int>(_Constant bool, _ExtVector<8, int>, _Constant bool, _ExtVector<8, int>, _ExtVector<8, int>, _Constant bool, _Constant bool, ...)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<16, int>, _ExtVector<16, int>, _Constant short, _ExtVector<8, __fp16>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<16, int>, _ExtVector<16, int>, _Constant short, _ExtVector<8, __fp16>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<16, int>, _ExtVector<16, int>, _Constant short, _ExtVector<8, __fp16>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<16, int>, _ExtVector<16, int>, _Constant short, _ExtVector<8, __fp16>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4 : AMDGPUBuiltin<"_ExtVector<8, float>(_Constant int, _ExtVector<16, int>, _Constant int, _ExtVector<16, int>, _Constant short, _ExtVector<8, float>)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<16, int>, _ExtVector<16, int>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<16, int>, _ExtVector<16, int>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<16, int>, _ExtVector<16, int>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<16, int>, _ExtVector<16, int>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4 : AMDGPUBuiltin<"_ExtVector<8, float>(_Constant int, _ExtVector<16, int>, _Constant int, _ExtVector<16, int>, _Constant short, _ExtVector<8, float>, _Constant int, _Constant int, int, _Constant int, _Constant int, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4 : AMDGPUBuiltin<"_ExtVector<8, float>(_Constant int, _ExtVector<16, int>, _Constant int, _ExtVector<16, int>, _Constant short, _ExtVector<8, float>, _Constant int, _Constant int, long int, _Constant int, _Constant int, long int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_16x16x32_f16 : AMDGPUBuiltin<"_ExtVector<8, float>(_Constant bool, _ExtVector<16, __fp16>, _Constant bool, _ExtVector<16, __fp16>, _Constant short, _ExtVector<8, float>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f16_16x16x32_f16 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_Constant bool, _ExtVector<16, __fp16>, _Constant bool, _ExtVector<16, __fp16>, _Constant short, _ExtVector<8, __fp16>, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_f32_32x16x128_f4 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<16, int>, _ExtVector<8, int>, _Constant short, _ExtVector<16, float>)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_scale_f32_32x16x128_f4 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<16, int>, _ExtVector<8, int>, _Constant short, _ExtVector<16, float>, _Constant int, _Constant int, int, _Constant int, _Constant int, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_wmma_scale16_f32_32x16x128_f4 : AMDGPUBuiltin<"_ExtVector<16, float>(_ExtVector<16, int>, _ExtVector<8, int>, _Constant short, _ExtVector<16, float>, _Constant int, _Constant int, long int, _Constant int, _Constant int, long int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f32_16x16x64_bf16 : AMDGPUBuiltin<"_ExtVector<8, float>(_Constant bool, _ExtVector<16, __bf16>, _Constant bool, _ExtVector<32, __bf16>, _ExtVector<8, float>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_bf16_16x16x64_bf16 : AMDGPUBuiltin<"_ExtVector<8, __bf16>(_Constant bool, _ExtVector<16, __bf16>, _Constant bool, _ExtVector<32, __bf16>, _ExtVector<8, __bf16>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16 : AMDGPUBuiltin<"_ExtVector<8, float>(_Constant bool, _ExtVector<16, __bf16>, _Constant bool, _ExtVector<32, __bf16>, _ExtVector<8, float>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, int>, _ExtVector<16, int>, _ExtVector<8, float>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f32_16x16x128_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, int>, _ExtVector<16, int>, _ExtVector<8, float>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f32_16x16x128_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, int>, _ExtVector<16, int>, _ExtVector<8, float>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f32_16x16x128_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<8, float>(_ExtVector<8, int>, _ExtVector<16, int>, _ExtVector<8, float>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f16_16x16x128_fp8_fp8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, int>, _ExtVector<16, int>, _ExtVector<8, __fp16>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f16_16x16x128_fp8_bf8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, int>, _ExtVector<16, int>, _ExtVector<8, __fp16>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f16_16x16x128_bf8_fp8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, int>, _ExtVector<16, int>, _ExtVector<8, __fp16>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f16_16x16x128_bf8_bf8 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_ExtVector<8, int>, _ExtVector<16, int>, _ExtVector<8, __fp16>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_i32_16x16x128_iu8 : AMDGPUBuiltin<"_ExtVector<8, int>(_Constant bool, _ExtVector<8, int>, _Constant bool, _ExtVector<16, int>, _ExtVector<8, int>, int, _Constant bool, _Constant bool, ...)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f32_16x16x64_f16 : AMDGPUBuiltin<"_ExtVector<8, float>(_Constant bool, _ExtVector<16, __fp16>, _Constant bool, _ExtVector<32, __fp16>, _ExtVector<8, float>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_swmmac_f16_16x16x64_f16 : AMDGPUBuiltin<"_ExtVector<8, __fp16>(_Constant bool, _ExtVector<16, __fp16>, _Constant bool, _ExtVector<32, __fp16>, _ExtVector<8, __fp16>, int, _Constant bool, _Constant bool)", [Const], "gfx1250-insts,wavefrontsize32">;

// GFX12.5 128B cooperative atomics
def __builtin_amdgcn_cooperative_atomic_load_32x4B : AMDGPUBuiltin<"int(int *, _Constant int, char const *)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_cooperative_atomic_store_32x4B : AMDGPUBuiltin<"void(int *, int, _Constant int, char const *)", [Const], "gfx1250-insts,wavefrontsize32">;

def __builtin_amdgcn_cooperative_atomic_load_16x8B : AMDGPUBuiltin<"_ExtVector<2, int>(_ExtVector<2, int *>, _Constant int, char const *)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_cooperative_atomic_store_16x8B : AMDGPUBuiltin<"void(_ExtVector<2, int *>, _ExtVector<2, int>, _Constant int, char const *)", [Const], "gfx1250-insts,wavefrontsize32">;

def __builtin_amdgcn_cooperative_atomic_load_8x16B : AMDGPUBuiltin<"_ExtVector<4, int>(_ExtVector<4, int *>, _Constant int, char const *)", [Const], "gfx1250-insts,wavefrontsize32">;
def __builtin_amdgcn_cooperative_atomic_store_8x16B : AMDGPUBuiltin<"void(_ExtVector<4, int *>, _ExtVector<4, int>, _Constant int, char const *)", [Const], "gfx1250-insts,wavefrontsize32">;

//===----------------------------------------------------------------------===//
// Image builtins
//===----------------------------------------------------------------------===//
def __builtin_amdgcn_image_load_1d_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_1d_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_1darray_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_1darray_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_2d_f32_i32 : AMDGPUBuiltin<"float(int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_2d_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_2d_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_2darray_f32_i32 : AMDGPUBuiltin<"float(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_2darray_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_2darray_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_3d_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_3d_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_cube_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_cube_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_1d_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_1d_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_1darray_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_1darray_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_2d_f32_i32 : AMDGPUBuiltin<"float(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_2d_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_2d_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_2darray_f32_i32 : AMDGPUBuiltin<"float(int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_2darray_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_2darray_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_3d_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_3d_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_cube_v4f32_i32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_load_mip_cube_v4f16_i32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_1d_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_1d_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_1darray_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_1darray_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_2d_f32_i32 : AMDGPUBuiltin<"void(float, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_2d_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_2d_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_2darray_f32_i32 : AMDGPUBuiltin<"void(float, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_2darray_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_2darray_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_3d_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_3d_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_cube_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_cube_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_1d_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_1d_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_1darray_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_1darray_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_2d_f32_i32 : AMDGPUBuiltin<"void(float, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_2d_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_2d_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_2darray_f32_i32 : AMDGPUBuiltin<"void(float, int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_2darray_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_2darray_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_3d_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_3d_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_cube_v4f32_i32 : AMDGPUBuiltin<"void(_ExtVector<4, float>, int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_store_mip_cube_v4f16_i32 : AMDGPUBuiltin<"void(_ExtVector<4, _Float16>, int, int, int, int, int, __amdgpu_texture_t, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_1d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_1d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_1darray_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_1darray_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_2d_f32_f32 : AMDGPUBuiltin<"float(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_2d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_2d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_2darray_f32_f32 : AMDGPUBuiltin<"float(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_2darray_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_2darray_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_3d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_3d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_cube_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_cube_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "image-insts">;
def __builtin_amdgcn_image_sample_lz_1d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_1d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_1darray_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_1darray_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_2d_f32_f32 : AMDGPUBuiltin<"float(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_2d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_2d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_2darray_f32_f32 : AMDGPUBuiltin<"float(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_2darray_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_2darray_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_3d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_3d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_cube_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_lz_cube_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_1d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_1d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_1darray_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_1darray_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_2d_f32_f32 : AMDGPUBuiltin<"float(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_2d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_2d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_2darray_f32_f32 : AMDGPUBuiltin<"float(int, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_2darray_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_2darray_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_3d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_3d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_cube_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_l_cube_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_1d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_1d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_1darray_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_1darray_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_2d_f32_f32 : AMDGPUBuiltin<"float(int, float, float, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_2d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_2d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_2darray_f32_f32 : AMDGPUBuiltin<"float(int, float, float, float, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_2darray_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_2darray_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_3d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, float, float, float, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_sample_d_3d_v4f16_f32 : AMDGPUBuiltin<"_ExtVector<4, _Float16>(int, float, float, float, float, float, float, float, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
def __builtin_amdgcn_image_gather4_lz_2d_v4f32_f32 : AMDGPUBuiltin<"_ExtVector<4, float>(int, float, float, __amdgpu_texture_t, _ExtVector<4, int>, bool, int, int)", [Const], "extended-image-insts">;
