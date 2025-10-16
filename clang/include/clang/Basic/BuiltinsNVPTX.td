//===--- BuiltinsNVPTX.td - NVPTX Builtin function defs ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PTX-specific builtin function database.
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

class SMFeatures {
  string Features;
}

class SM<string version, list<SMFeatures> newer_list> : SMFeatures {
  let Features = !foldl(!strconcat("sm_", version), newer_list, f, newer,
                        !strconcat(f, "|", newer.Features));
}

let Features = "sm_121a" in def SM_121a : SMFeatures;
let Features = "sm_120a" in def SM_120a : SMFeatures;
let Features = "sm_103a" in def SM_103a : SMFeatures;
let Features = "sm_101a" in def SM_101a : SMFeatures;
let Features = "sm_100a" in def SM_100a : SMFeatures;
let Features = "sm_90a" in def SM_90a : SMFeatures;

def SM_121 : SM<"121", [SM_121a]>;
def SM_120 : SM<"120", [SM_120a, SM_121]>;
def SM_103 : SM<"103", [SM_103a, SM_120]>;
def SM_101 : SM<"101", [SM_101a, SM_103]>;
def SM_100 : SM<"100", [SM_100a, SM_101]>;
def SM_90 : SM<"90", [SM_90a, SM_100]>;
def SM_89 : SM<"89", [SM_90]>;
def SM_87 : SM<"87", [SM_89]>;
def SM_86 : SM<"86", [SM_87]>;
def SM_80 : SM<"80", [SM_86]>;
def SM_75 : SM<"75", [SM_80]>;
def SM_72 : SM<"72", [SM_75]>;
def SM_70 : SM<"70", [SM_72]>;
def SM_62 : SM<"62", [SM_70]>;
def SM_61 : SM<"61", [SM_62]>;
def SM_60 : SM<"60", [SM_61]>;
def SM_53 : SM<"53", [SM_60]>;

class PTXFeatures {
  string Features;
}

class PTX<string version, PTXFeatures newer> : PTXFeatures {
  let Features = !strconcat("ptx", version, "|", newer.Features);
}

let Features = "ptx88" in def PTX88 : PTXFeatures;

def PTX87 : PTX<"87", PTX88>;
def PTX86 : PTX<"86", PTX87>;
def PTX85 : PTX<"85", PTX86>;
def PTX84 : PTX<"84", PTX85>;
def PTX83 : PTX<"83", PTX84>;
def PTX82 : PTX<"82", PTX83>;
def PTX81 : PTX<"81", PTX82>;
def PTX80 : PTX<"80", PTX81>;
def PTX78 : PTX<"78", PTX80>;
def PTX77 : PTX<"77", PTX78>;
def PTX76 : PTX<"76", PTX77>;
def PTX75 : PTX<"75", PTX76>;
def PTX74 : PTX<"74", PTX75>;
def PTX73 : PTX<"73", PTX74>;
def PTX72 : PTX<"72", PTX73>;
def PTX71 : PTX<"71", PTX72>;
def PTX70 : PTX<"70", PTX71>;
def PTX65 : PTX<"65", PTX70>;
def PTX64 : PTX<"64", PTX65>;
def PTX63 : PTX<"63", PTX64>;
def PTX62 : PTX<"62", PTX63>;
def PTX61 : PTX<"61", PTX62>;
def PTX60 : PTX<"60", PTX61>;
def PTX42 : PTX<"42", PTX60>;

class NVPTXBuiltin<string prototype> : TargetBuiltin {
  let Spellings = [NAME];
  let Prototype = prototype;
}

class NVPTXBuiltinSM<string prototype, SMFeatures sm> : NVPTXBuiltin<prototype> {
  let Features = sm.Features;
}

class NVPTXBuiltinPTX<string prototype, PTXFeatures ptx> : NVPTXBuiltin<prototype> {
  let Features = ptx.Features;
}

class NVPTXBuiltinSMAndPTX<string prototype, SMFeatures sm, PTXFeatures ptx> : NVPTXBuiltin<prototype> {
  let Features = !strconcat("(", sm.Features, "),(", ptx.Features, ")");
}

// Special Registers

let Attributes = [NoThrow, Const] in {
  def __nvvm_read_ptx_sreg_tid_x : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_tid_y : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_tid_z : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_tid_w : NVPTXBuiltin<"int()">;

  def __nvvm_read_ptx_sreg_ntid_x : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_ntid_y : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_ntid_z : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_ntid_w : NVPTXBuiltin<"int()">;

  def __nvvm_read_ptx_sreg_ctaid_x : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_ctaid_y : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_ctaid_z : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_ctaid_w : NVPTXBuiltin<"int()">;

  def __nvvm_read_ptx_sreg_nctaid_x : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_nctaid_y : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_nctaid_z : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_nctaid_w : NVPTXBuiltin<"int()">;

  def __nvvm_read_ptx_sreg_clusterid_x : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_clusterid_y : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_clusterid_z : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_clusterid_w : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;

  def __nvvm_read_ptx_sreg_nclusterid_x : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_nclusterid_y : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_nclusterid_z : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_nclusterid_w : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;

  def __nvvm_read_ptx_sreg_cluster_ctaid_x : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_cluster_ctaid_y : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_cluster_ctaid_z : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_cluster_ctaid_w : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;

  def __nvvm_read_ptx_sreg_cluster_nctaid_x : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_cluster_nctaid_y : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_cluster_nctaid_z : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_cluster_nctaid_w : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;

  def __nvvm_read_ptx_sreg_cluster_ctarank : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;
  def __nvvm_read_ptx_sreg_cluster_nctarank : NVPTXBuiltinSMAndPTX<"int()", SM_90, PTX78>;

  def __nvvm_is_explicit_cluster : NVPTXBuiltinSMAndPTX<"bool()", SM_90, PTX78>;

  def __nvvm_read_ptx_sreg_laneid : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_warpid : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_nwarpid : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_warpsize : NVPTXBuiltin<"int()">;

  def __nvvm_read_ptx_sreg_smid : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_nsmid : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_gridid : NVPTXBuiltin<"int()">;

  def __nvvm_read_ptx_sreg_lanemask_eq : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_lanemask_le : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_lanemask_lt : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_lanemask_ge : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_lanemask_gt : NVPTXBuiltin<"int()">;
}

let Attributes = [NoThrow] in {
  def __nvvm_read_ptx_sreg_clock : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_clock64 : NVPTXBuiltin<"long long int()">;
  def __nvvm_read_ptx_sreg_globaltimer : NVPTXBuiltin<"long long int()">;

  def __nvvm_read_ptx_sreg_pm0 : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_pm1 : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_pm2 : NVPTXBuiltin<"int()">;
  def __nvvm_read_ptx_sreg_pm3 : NVPTXBuiltin<"int()">;
}

// MISC

def __nvvm_prmt : NVPTXBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)">;
let Attributes = [NoReturn] in {
  def __nvvm_exit : NVPTXBuiltin<"void()">;
  def __nvvm_reflect : NVPTXBuiltin<"unsigned int(char const *)">;
}
let Attributes = [NoThrow] in {
  def __nvvm_nanosleep : NVPTXBuiltinSMAndPTX<"void(unsigned int)", SM_70, PTX63>;
  def __nvvm_pm_event_mask : NVPTXBuiltin<"void(_Constant unsigned short)">;
}

// Min Max

def __nvvm_fmin_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_80, PTX70>;
def __nvvm_fmin_ftz_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_80, PTX70>;
def __nvvm_fmin_nan_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_80, PTX70>;
def __nvvm_fmin_ftz_nan_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_80, PTX70>;
def __nvvm_fmin_xorsign_abs_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_86, PTX72>;
def __nvvm_fmin_ftz_xorsign_abs_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_86, PTX72>;
def __nvvm_fmin_nan_xorsign_abs_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_86, PTX72>;
def __nvvm_fmin_ftz_nan_xorsign_abs_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_86, PTX72>;
def __nvvm_fmin_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_80, PTX70>;
def __nvvm_fmin_ftz_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_80, PTX70>;
def __nvvm_fmin_nan_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_80, PTX70>;
def __nvvm_fmin_ftz_nan_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_80, PTX70>;
def __nvvm_fmin_xorsign_abs_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_86, PTX72>;
def __nvvm_fmin_ftz_xorsign_abs_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_86, PTX72>;
def __nvvm_fmin_nan_xorsign_abs_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_86, PTX72>;
def __nvvm_fmin_ftz_nan_xorsign_abs_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_86, PTX72>;
def __nvvm_fmin_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_80, PTX70>;
def __nvvm_fmin_ftz_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_80, PTX70>;
def __nvvm_fmin_nan_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_80, PTX70>;
def __nvvm_fmin_ftz_nan_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_80, PTX70>;
def __nvvm_fmin_xorsign_abs_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_86, PTX72>;
def __nvvm_fmin_nan_xorsign_abs_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_86, PTX72>;
def __nvvm_fmin_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_80, PTX70>;
def __nvvm_fmin_ftz_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_80, PTX70>;
def __nvvm_fmin_nan_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_80, PTX70>;
def __nvvm_fmin_ftz_nan_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_80, PTX70>;
def __nvvm_fmin_xorsign_abs_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_86, PTX72>;
def __nvvm_fmin_nan_xorsign_abs_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_86, PTX72>;
def __nvvm_fmin_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_fmin_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_fmin_nan_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_80, PTX70>;
def __nvvm_fmin_ftz_nan_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_80, PTX70>;
def __nvvm_fmin_xorsign_abs_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_86, PTX72>;
def __nvvm_fmin_ftz_xorsign_abs_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_86, PTX72>;
def __nvvm_fmin_nan_xorsign_abs_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_86, PTX72>;
def __nvvm_fmin_ftz_nan_xorsign_abs_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_86, PTX72>;
def __nvvm_fmin_d : NVPTXBuiltin<"double(double, double)">;

def __nvvm_fmax_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_80, PTX70>;
def __nvvm_fmax_ftz_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_80, PTX70>;
def __nvvm_fmax_nan_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_80, PTX70>;
def __nvvm_fmax_ftz_nan_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_80, PTX70>;
def __nvvm_fmax_xorsign_abs_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_86, PTX72>;
def __nvvm_fmax_ftz_xorsign_abs_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_86, PTX72>;
def __nvvm_fmax_nan_xorsign_abs_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_86, PTX72>;
def __nvvm_fmax_ftz_nan_xorsign_abs_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16)", SM_86, PTX72>;
def __nvvm_fmax_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_80, PTX70>;
def __nvvm_fmax_ftz_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_80, PTX70>;
def __nvvm_fmax_nan_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_80, PTX70>;
def __nvvm_fmax_ftz_nan_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_80, PTX70>;
def __nvvm_fmax_xorsign_abs_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_86, PTX72>;
def __nvvm_fmax_ftz_xorsign_abs_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_86, PTX72>;
def __nvvm_fmax_nan_xorsign_abs_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_86, PTX72>;
def __nvvm_fmax_ftz_nan_xorsign_abs_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>)", SM_86, PTX72>;
def __nvvm_fmax_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_80, PTX70>;
def __nvvm_fmax_ftz_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_80, PTX70>;
def __nvvm_fmax_nan_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_80, PTX70>;
def __nvvm_fmax_ftz_nan_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_80, PTX70>;
def __nvvm_fmax_xorsign_abs_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_86, PTX72>;
def __nvvm_fmax_nan_xorsign_abs_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16)", SM_86, PTX72>;
def __nvvm_fmax_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_80, PTX70>;
def __nvvm_fmax_ftz_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_80, PTX70>;
def __nvvm_fmax_nan_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_80, PTX70>;
def __nvvm_fmax_ftz_nan_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_80, PTX70>;
def __nvvm_fmax_xorsign_abs_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_86, PTX72>;
def __nvvm_fmax_nan_xorsign_abs_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>)", SM_86, PTX72>;
def __nvvm_fmax_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_fmax_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_fmax_nan_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_80, PTX70>;
def __nvvm_fmax_ftz_nan_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_80, PTX70>;
def __nvvm_fmax_xorsign_abs_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_86, PTX72>;
def __nvvm_fmax_ftz_xorsign_abs_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_86, PTX72>;
def __nvvm_fmax_nan_xorsign_abs_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_86, PTX72>;
def __nvvm_fmax_ftz_nan_xorsign_abs_f : NVPTXBuiltinSMAndPTX<"float(float, float)", SM_86, PTX72>;
def __nvvm_fmax_d : NVPTXBuiltin<"double(double, double)">;

// Multiplication

def __nvvm_mulhi_i : NVPTXBuiltin<"int(int, int)">;
def __nvvm_mulhi_ui : NVPTXBuiltin<"unsigned int(unsigned int, unsigned int)">;
def __nvvm_mulhi_ll : NVPTXBuiltin<"long long int(long long int, long long int)">;
def __nvvm_mulhi_ull : NVPTXBuiltin<"unsigned long long int(unsigned long long int, unsigned long long int)">;

def __nvvm_mul_rn_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_mul_rn_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_mul_rz_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_mul_rz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_mul_rm_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_mul_rm_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_mul_rp_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_mul_rp_f : NVPTXBuiltin<"float(float, float)">;

def __nvvm_mul_rn_d : NVPTXBuiltin<"double(double, double)">;
def __nvvm_mul_rz_d : NVPTXBuiltin<"double(double, double)">;
def __nvvm_mul_rm_d : NVPTXBuiltin<"double(double, double)">;
def __nvvm_mul_rp_d : NVPTXBuiltin<"double(double, double)">;

def __nvvm_mul24_i : NVPTXBuiltin<"int(int, int)">;
def __nvvm_mul24_ui : NVPTXBuiltin<"unsigned int(unsigned int, unsigned int)">;

// Div

def __nvvm_div_approx_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_div_approx_f : NVPTXBuiltin<"float(float, float)">;

def __nvvm_div_rn_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_div_rn_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_div_rz_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_div_rz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_div_rm_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_div_rm_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_div_rp_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_div_rp_f : NVPTXBuiltin<"float(float, float)">;

def __nvvm_div_rn_d : NVPTXBuiltin<"double(double, double)">;
def __nvvm_div_rz_d : NVPTXBuiltin<"double(double, double)">;
def __nvvm_div_rm_d : NVPTXBuiltin<"double(double, double)">;
def __nvvm_div_rp_d : NVPTXBuiltin<"double(double, double)">;

// Sad

def __nvvm_sad_i : NVPTXBuiltin<"int(int, int, int)">;
def __nvvm_sad_ui : NVPTXBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)">;

// Floor, Ceil

def __nvvm_floor_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_floor_f : NVPTXBuiltin<"float(float)">;
def __nvvm_floor_d : NVPTXBuiltin<"double(double)">;

def __nvvm_ceil_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_ceil_f : NVPTXBuiltin<"float(float)">;
def __nvvm_ceil_d : NVPTXBuiltin<"double(double)">;

// Abs

def __nvvm_fabs_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_fabs_f : NVPTXBuiltin<"float(float)">;
def __nvvm_fabs_d : NVPTXBuiltin<"double(double)">;

def __nvvm_fabs_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16)", SM_53, PTX65>;
def __nvvm_fabs_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>)", SM_53, PTX65>;
def __nvvm_fabs_ftz_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16)", SM_53, PTX65>;
def __nvvm_fabs_ftz_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>)", SM_53, PTX65>;

// Round

def __nvvm_round_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_round_f : NVPTXBuiltin<"float(float)">;
def __nvvm_round_d : NVPTXBuiltin<"double(double)">;

// Trunc

def __nvvm_trunc_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_trunc_f : NVPTXBuiltin<"float(float)">;
def __nvvm_trunc_d : NVPTXBuiltin<"double(double)">;

// Saturate

def __nvvm_saturate_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_saturate_f : NVPTXBuiltin<"float(float)">;
def __nvvm_saturate_d : NVPTXBuiltin<"double(double)">;

// Exp2, Log2

def __nvvm_ex2_approx_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_ex2_approx_f : NVPTXBuiltin<"float(float)">;
def __nvvm_ex2_approx_d : NVPTXBuiltin<"double(double)">;
def __nvvm_ex2_approx_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16)", SM_75, PTX70>;
def __nvvm_ex2_approx_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>)", SM_75, PTX70>;

def __nvvm_lg2_approx_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_lg2_approx_f : NVPTXBuiltin<"float(float)">;
def __nvvm_lg2_approx_d : NVPTXBuiltin<"double(double)">;

// Sin, Cos

def __nvvm_sin_approx_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_sin_approx_f : NVPTXBuiltin<"float(float)">;

def __nvvm_cos_approx_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_cos_approx_f : NVPTXBuiltin<"float(float)">;

// Fma

def __nvvm_fma_rn_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16, __fp16)", SM_53, PTX42>;
def __nvvm_fma_rn_ftz_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16, __fp16)", SM_53, PTX42>;
def __nvvm_fma_rn_sat_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16, __fp16)", SM_53, PTX42>;
def __nvvm_fma_rn_ftz_sat_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16, __fp16)", SM_53, PTX42>;
def __nvvm_fma_rn_relu_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16, __fp16)", SM_80, PTX70>;
def __nvvm_fma_rn_ftz_relu_f16 : NVPTXBuiltinSMAndPTX<"__fp16(__fp16, __fp16, __fp16)", SM_80, PTX70>;
def __nvvm_fma_rn_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>, _Vector<2, __fp16>)", SM_53, PTX42>;
def __nvvm_fma_rn_ftz_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>, _Vector<2, __fp16>)", SM_53, PTX42>;
def __nvvm_fma_rn_sat_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>, _Vector<2, __fp16>)", SM_53, PTX42>;
def __nvvm_fma_rn_ftz_sat_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>, _Vector<2, __fp16>)", SM_53, PTX42>;
def __nvvm_fma_rn_relu_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>, _Vector<2, __fp16>)", SM_80, PTX70>;
def __nvvm_fma_rn_ftz_relu_f16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(_Vector<2, __fp16>, _Vector<2, __fp16>, _Vector<2, __fp16>)", SM_80, PTX70>;
def __nvvm_fma_rn_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16, __bf16)", SM_80, PTX70>;
def __nvvm_fma_rn_relu_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16, __bf16, __bf16)", SM_80, PTX70>;
def __nvvm_fma_rn_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>, _Vector<2, __bf16>)", SM_80, PTX70>;
def __nvvm_fma_rn_relu_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>, _Vector<2, __bf16>, _Vector<2, __bf16>)", SM_80, PTX70>;
def __nvvm_fma_rn_ftz_f : NVPTXBuiltin<"float(float, float, float)">;
def __nvvm_fma_rn_f : NVPTXBuiltin<"float(float, float, float)">;
def __nvvm_fma_rz_ftz_f : NVPTXBuiltin<"float(float, float, float)">;
def __nvvm_fma_rz_f : NVPTXBuiltin<"float(float, float, float)">;
def __nvvm_fma_rm_ftz_f : NVPTXBuiltin<"float(float, float, float)">;
def __nvvm_fma_rm_f : NVPTXBuiltin<"float(float, float, float)">;
def __nvvm_fma_rp_ftz_f : NVPTXBuiltin<"float(float, float, float)">;
def __nvvm_fma_rp_f : NVPTXBuiltin<"float(float, float, float)">;
def __nvvm_fma_rn_d : NVPTXBuiltin<"double(double, double, double)">;
def __nvvm_fma_rz_d : NVPTXBuiltin<"double(double, double, double)">;
def __nvvm_fma_rm_d : NVPTXBuiltin<"double(double, double, double)">;
def __nvvm_fma_rp_d : NVPTXBuiltin<"double(double, double, double)">;

// Rcp

def __nvvm_rcp_rn_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_rcp_rn_f : NVPTXBuiltin<"float(float)">;
def __nvvm_rcp_rz_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_rcp_rz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_rcp_rm_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_rcp_rm_f : NVPTXBuiltin<"float(float)">;
def __nvvm_rcp_rp_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_rcp_rp_f : NVPTXBuiltin<"float(float)">;

def __nvvm_rcp_rn_d : NVPTXBuiltin<"double(double)">;
def __nvvm_rcp_rz_d : NVPTXBuiltin<"double(double)">;
def __nvvm_rcp_rm_d : NVPTXBuiltin<"double(double)">;
def __nvvm_rcp_rp_d : NVPTXBuiltin<"double(double)">;

def __nvvm_rcp_approx_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_rcp_approx_ftz_d : NVPTXBuiltin<"double(double)">;

// Sqrt

def __nvvm_sqrt_rn_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_sqrt_rn_f : NVPTXBuiltin<"float(float)">;
def __nvvm_sqrt_rz_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_sqrt_rz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_sqrt_rm_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_sqrt_rm_f : NVPTXBuiltin<"float(float)">;
def __nvvm_sqrt_rp_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_sqrt_rp_f : NVPTXBuiltin<"float(float)">;
def __nvvm_sqrt_approx_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_sqrt_approx_f : NVPTXBuiltin<"float(float)">;

def __nvvm_sqrt_rn_d : NVPTXBuiltin<"double(double)">;
def __nvvm_sqrt_rz_d : NVPTXBuiltin<"double(double)">;
def __nvvm_sqrt_rm_d : NVPTXBuiltin<"double(double)">;
def __nvvm_sqrt_rp_d : NVPTXBuiltin<"double(double)">;

// Rsqrt

def __nvvm_rsqrt_approx_ftz_f : NVPTXBuiltin<"float(float)">;
def __nvvm_rsqrt_approx_f : NVPTXBuiltin<"float(float)">;
def __nvvm_rsqrt_approx_d : NVPTXBuiltin<"double(double)">;

// Add

def __nvvm_add_rn_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_add_rn_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_add_rz_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_add_rz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_add_rm_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_add_rm_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_add_rp_ftz_f : NVPTXBuiltin<"float(float, float)">;
def __nvvm_add_rp_f : NVPTXBuiltin<"float(float, float)">;

def __nvvm_add_rn_d : NVPTXBuiltin<"double(double, double)">;
def __nvvm_add_rz_d : NVPTXBuiltin<"double(double, double)">;
def __nvvm_add_rm_d : NVPTXBuiltin<"double(double, double)">;
def __nvvm_add_rp_d : NVPTXBuiltin<"double(double, double)">;

// Convert

def __nvvm_d2f_rn_ftz : NVPTXBuiltin<"float(double)">;
def __nvvm_d2f_rn : NVPTXBuiltin<"float(double)">;
def __nvvm_d2f_rz_ftz : NVPTXBuiltin<"float(double)">;
def __nvvm_d2f_rz : NVPTXBuiltin<"float(double)">;
def __nvvm_d2f_rm_ftz : NVPTXBuiltin<"float(double)">;
def __nvvm_d2f_rm : NVPTXBuiltin<"float(double)">;
def __nvvm_d2f_rp_ftz : NVPTXBuiltin<"float(double)">;
def __nvvm_d2f_rp : NVPTXBuiltin<"float(double)">;

def __nvvm_d2i_rn : NVPTXBuiltin<"int(double)">;
def __nvvm_d2i_rz : NVPTXBuiltin<"int(double)">;
def __nvvm_d2i_rm : NVPTXBuiltin<"int(double)">;
def __nvvm_d2i_rp : NVPTXBuiltin<"int(double)">;

def __nvvm_d2ui_rn : NVPTXBuiltin<"unsigned int(double)">;
def __nvvm_d2ui_rz : NVPTXBuiltin<"unsigned int(double)">;
def __nvvm_d2ui_rm : NVPTXBuiltin<"unsigned int(double)">;
def __nvvm_d2ui_rp : NVPTXBuiltin<"unsigned int(double)">;

def __nvvm_i2d_rn : NVPTXBuiltin<"double(int)">;
def __nvvm_i2d_rz : NVPTXBuiltin<"double(int)">;
def __nvvm_i2d_rm : NVPTXBuiltin<"double(int)">;
def __nvvm_i2d_rp : NVPTXBuiltin<"double(int)">;

def __nvvm_ui2d_rn : NVPTXBuiltin<"double(unsigned int)">;
def __nvvm_ui2d_rz : NVPTXBuiltin<"double(unsigned int)">;
def __nvvm_ui2d_rm : NVPTXBuiltin<"double(unsigned int)">;
def __nvvm_ui2d_rp : NVPTXBuiltin<"double(unsigned int)">;

def __nvvm_f2i_rn_ftz : NVPTXBuiltin<"int(float)">;
def __nvvm_f2i_rn : NVPTXBuiltin<"int(float)">;
def __nvvm_f2i_rz_ftz : NVPTXBuiltin<"int(float)">;
def __nvvm_f2i_rz : NVPTXBuiltin<"int(float)">;
def __nvvm_f2i_rm_ftz : NVPTXBuiltin<"int(float)">;
def __nvvm_f2i_rm : NVPTXBuiltin<"int(float)">;
def __nvvm_f2i_rp_ftz : NVPTXBuiltin<"int(float)">;
def __nvvm_f2i_rp : NVPTXBuiltin<"int(float)">;

def __nvvm_f2ui_rn_ftz : NVPTXBuiltin<"unsigned int(float)">;
def __nvvm_f2ui_rn : NVPTXBuiltin<"unsigned int(float)">;
def __nvvm_f2ui_rz_ftz : NVPTXBuiltin<"unsigned int(float)">;
def __nvvm_f2ui_rz : NVPTXBuiltin<"unsigned int(float)">;
def __nvvm_f2ui_rm_ftz : NVPTXBuiltin<"unsigned int(float)">;
def __nvvm_f2ui_rm : NVPTXBuiltin<"unsigned int(float)">;
def __nvvm_f2ui_rp_ftz : NVPTXBuiltin<"unsigned int(float)">;
def __nvvm_f2ui_rp : NVPTXBuiltin<"unsigned int(float)">;

def __nvvm_i2f_rn : NVPTXBuiltin<"float(int)">;
def __nvvm_i2f_rz : NVPTXBuiltin<"float(int)">;
def __nvvm_i2f_rm : NVPTXBuiltin<"float(int)">;
def __nvvm_i2f_rp : NVPTXBuiltin<"float(int)">;

def __nvvm_ui2f_rn : NVPTXBuiltin<"float(unsigned int)">;
def __nvvm_ui2f_rz : NVPTXBuiltin<"float(unsigned int)">;
def __nvvm_ui2f_rm : NVPTXBuiltin<"float(unsigned int)">;
def __nvvm_ui2f_rp : NVPTXBuiltin<"float(unsigned int)">;

def __nvvm_lohi_i2d : NVPTXBuiltin<"double(int, int)">;

def __nvvm_d2i_lo : NVPTXBuiltin<"int(double)">;
def __nvvm_d2i_hi : NVPTXBuiltin<"int(double)">;

def __nvvm_f2ll_rn_ftz : NVPTXBuiltin<"long long int(float)">;
def __nvvm_f2ll_rn : NVPTXBuiltin<"long long int(float)">;
def __nvvm_f2ll_rz_ftz : NVPTXBuiltin<"long long int(float)">;
def __nvvm_f2ll_rz : NVPTXBuiltin<"long long int(float)">;
def __nvvm_f2ll_rm_ftz : NVPTXBuiltin<"long long int(float)">;
def __nvvm_f2ll_rm : NVPTXBuiltin<"long long int(float)">;
def __nvvm_f2ll_rp_ftz : NVPTXBuiltin<"long long int(float)">;
def __nvvm_f2ll_rp : NVPTXBuiltin<"long long int(float)">;

def __nvvm_f2ull_rn_ftz : NVPTXBuiltin<"unsigned long long int(float)">;
def __nvvm_f2ull_rn : NVPTXBuiltin<"unsigned long long int(float)">;
def __nvvm_f2ull_rz_ftz : NVPTXBuiltin<"unsigned long long int(float)">;
def __nvvm_f2ull_rz : NVPTXBuiltin<"unsigned long long int(float)">;
def __nvvm_f2ull_rm_ftz : NVPTXBuiltin<"unsigned long long int(float)">;
def __nvvm_f2ull_rm : NVPTXBuiltin<"unsigned long long int(float)">;
def __nvvm_f2ull_rp_ftz : NVPTXBuiltin<"unsigned long long int(float)">;
def __nvvm_f2ull_rp : NVPTXBuiltin<"unsigned long long int(float)">;

def __nvvm_d2ll_rn : NVPTXBuiltin<"long long int(double)">;
def __nvvm_d2ll_rz : NVPTXBuiltin<"long long int(double)">;
def __nvvm_d2ll_rm : NVPTXBuiltin<"long long int(double)">;
def __nvvm_d2ll_rp : NVPTXBuiltin<"long long int(double)">;

def __nvvm_d2ull_rn : NVPTXBuiltin<"unsigned long long int(double)">;
def __nvvm_d2ull_rz : NVPTXBuiltin<"unsigned long long int(double)">;
def __nvvm_d2ull_rm : NVPTXBuiltin<"unsigned long long int(double)">;
def __nvvm_d2ull_rp : NVPTXBuiltin<"unsigned long long int(double)">;

def __nvvm_ll2f_rn : NVPTXBuiltin<"float(long long int)">;
def __nvvm_ll2f_rz : NVPTXBuiltin<"float(long long int)">;
def __nvvm_ll2f_rm : NVPTXBuiltin<"float(long long int)">;
def __nvvm_ll2f_rp : NVPTXBuiltin<"float(long long int)">;

def __nvvm_ull2f_rn : NVPTXBuiltin<"float(unsigned long long int)">;
def __nvvm_ull2f_rz : NVPTXBuiltin<"float(unsigned long long int)">;
def __nvvm_ull2f_rm : NVPTXBuiltin<"float(unsigned long long int)">;
def __nvvm_ull2f_rp : NVPTXBuiltin<"float(unsigned long long int)">;

def __nvvm_ll2d_rn : NVPTXBuiltin<"double(long long int)">;
def __nvvm_ll2d_rz : NVPTXBuiltin<"double(long long int)">;
def __nvvm_ll2d_rm : NVPTXBuiltin<"double(long long int)">;
def __nvvm_ll2d_rp : NVPTXBuiltin<"double(long long int)">;

def __nvvm_ull2d_rn : NVPTXBuiltin<"double(unsigned long long int)">;
def __nvvm_ull2d_rz : NVPTXBuiltin<"double(unsigned long long int)">;
def __nvvm_ull2d_rm : NVPTXBuiltin<"double(unsigned long long int)">;
def __nvvm_ull2d_rp : NVPTXBuiltin<"double(unsigned long long int)">;

def __nvvm_f2h_rn_ftz : NVPTXBuiltin<"unsigned short(float)">;
def __nvvm_f2h_rn : NVPTXBuiltin<"unsigned short(float)">;

def __nvvm_ff2bf16x2_rn : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(float, float)", SM_80, PTX70>;
def __nvvm_ff2bf16x2_rn_relu : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(float, float)", SM_80, PTX70>;
def __nvvm_ff2bf16x2_rz : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(float, float)", SM_80, PTX70>;
def __nvvm_ff2bf16x2_rz_relu : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(float, float)", SM_80, PTX70>;
def __nvvm_ff2bf16x2_rs :
  NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(float, float, uint32_t)",
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_ff2bf16x2_rs_relu :
  NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(float, float, uint32_t)",
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_ff2bf16x2_rs_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(float, float, uint32_t)",
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_ff2bf16x2_rs_relu_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(float, float, uint32_t)",
                       SM<"100a", [SM_103a]>, PTX87>;

def __nvvm_ff2f16x2_rn : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(float, float)", SM_80, PTX70>;
def __nvvm_ff2f16x2_rn_relu : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(float, float)", SM_80, PTX70>;
def __nvvm_ff2f16x2_rz : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(float, float)", SM_80, PTX70>;
def __nvvm_ff2f16x2_rz_relu : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(float, float)", SM_80, PTX70>;
def __nvvm_ff2f16x2_rs :
  NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(float, float, uint32_t)",
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_ff2f16x2_rs_relu :
  NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(float, float, uint32_t)",
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_ff2f16x2_rs_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(float, float, uint32_t)",
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_ff2f16x2_rs_relu_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(float, float, uint32_t)",
                       SM<"100a", [SM_103a]>, PTX87>;

def __nvvm_f2bf16_rn : NVPTXBuiltinSMAndPTX<"__bf16(float)", SM_80, PTX70>;
def __nvvm_f2bf16_rn_relu : NVPTXBuiltinSMAndPTX<"__bf16(float)", SM_80, PTX70>;
def __nvvm_f2bf16_rz : NVPTXBuiltinSMAndPTX<"__bf16(float)", SM_80, PTX70>;
def __nvvm_f2bf16_rz_relu : NVPTXBuiltinSMAndPTX<"__bf16(float)", SM_80, PTX70>;

def __nvvm_f2tf32_rna : NVPTXBuiltinSMAndPTX<"int32_t(float)", SM_80, PTX70>;
def __nvvm_f2tf32_rna_satfinite : NVPTXBuiltinSMAndPTX<"int32_t(float)", SM_89, PTX81>;
def __nvvm_f2tf32_rn : NVPTXBuiltinSMAndPTX<"int32_t(float)", SM_90, PTX78>;
def __nvvm_f2tf32_rn_relu : NVPTXBuiltinSMAndPTX<"int32_t(float)", SM_90, PTX78>;
def __nvvm_f2tf32_rn_satfinite : NVPTXBuiltinSMAndPTX<"int32_t(float)", SM_100, PTX86>;
def __nvvm_f2tf32_rn_relu_satfinite : NVPTXBuiltinSMAndPTX<"int32_t(float)", SM_100, PTX86>;
def __nvvm_f2tf32_rz : NVPTXBuiltinSMAndPTX<"int32_t(float)", SM_90, PTX78>;
def __nvvm_f2tf32_rz_relu : NVPTXBuiltinSMAndPTX<"int32_t(float)", SM_90, PTX78>;
def __nvvm_f2tf32_rz_satfinite : NVPTXBuiltinSMAndPTX<"int32_t(float)", SM_100, PTX86>;
def __nvvm_f2tf32_rz_relu_satfinite : NVPTXBuiltinSMAndPTX<"int32_t(float)", SM_100, PTX86>;

def __nvvm_ff_to_e4m3x2_rn : NVPTXBuiltinSMAndPTX<"short(float, float)", SM_89, PTX81>;
def __nvvm_ff_to_e4m3x2_rn_relu : NVPTXBuiltinSMAndPTX<"short(float, float)", SM_89, PTX81>;
def __nvvm_ff_to_e5m2x2_rn : NVPTXBuiltinSMAndPTX<"short(float, float)", SM_89, PTX81>;
def __nvvm_ff_to_e5m2x2_rn_relu : NVPTXBuiltinSMAndPTX<"short(float, float)", SM_89, PTX81>;

def __nvvm_f16x2_to_e4m3x2_rn : NVPTXBuiltinSMAndPTX<"short(_Vector<2, __fp16>)", SM_89, PTX81>;
def __nvvm_f16x2_to_e4m3x2_rn_relu : NVPTXBuiltinSMAndPTX<"short(_Vector<2, __fp16>)", SM_89, PTX81>;
def __nvvm_f16x2_to_e5m2x2_rn : NVPTXBuiltinSMAndPTX<"short(_Vector<2, __fp16>)", SM_89, PTX81>;
def __nvvm_f16x2_to_e5m2x2_rn_relu : NVPTXBuiltinSMAndPTX<"short(_Vector<2, __fp16>)", SM_89, PTX81>;

def __nvvm_e4m3x2_to_f16x2_rn : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(short)", SM_89, PTX81>;
def __nvvm_e4m3x2_to_f16x2_rn_relu : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(short)", SM_89, PTX81>;
def __nvvm_e5m2x2_to_f16x2_rn : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(short)", SM_89, PTX81>;
def __nvvm_e5m2x2_to_f16x2_rn_relu : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(short)", SM_89, PTX81>;

def __nvvm_f32x4_to_e4m3x4_rs_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<4, char>(_Vector<4, float>, uint32_t)", 
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_f32x4_to_e4m3x4_rs_relu_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<4, char>(_Vector<4, float>, uint32_t)", 
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_f32x4_to_e5m2x4_rs_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<4, char>(_Vector<4, float>, uint32_t)", 
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_f32x4_to_e5m2x4_rs_relu_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<4, char>(_Vector<4, float>, uint32_t)", 
                       SM<"100a", [SM_103a]>, PTX87>;

def __nvvm_ff_to_e2m3x2_rn_satfinite : NVPTXBuiltinSMAndPTX<"short(float, float)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_ff_to_e2m3x2_rn_relu_satfinite : NVPTXBuiltinSMAndPTX<"short(float, float)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_ff_to_e3m2x2_rn_satfinite : NVPTXBuiltinSMAndPTX<"short(float, float)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_ff_to_e3m2x2_rn_relu_satfinite : NVPTXBuiltinSMAndPTX<"short(float, float)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;

def __nvvm_e2m3x2_to_f16x2_rn : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(short)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_e2m3x2_to_f16x2_rn_relu : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(short)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_e3m2x2_to_f16x2_rn : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(short)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_e3m2x2_to_f16x2_rn_relu : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(short)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;

def __nvvm_f32x4_to_e2m3x4_rs_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<4, char>(_Vector<4, float>, uint32_t)", 
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_f32x4_to_e2m3x4_rs_relu_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<4, char>(_Vector<4, float>, uint32_t)", 
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_f32x4_to_e3m2x4_rs_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<4, char>(_Vector<4, float>, uint32_t)", 
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_f32x4_to_e3m2x4_rs_relu_satfinite :
  NVPTXBuiltinSMAndPTX<"_Vector<4, char>(_Vector<4, float>, uint32_t)", 
                       SM<"100a", [SM_103a]>, PTX87>;

def __nvvm_ff_to_e2m1x2_rn_satfinite : NVPTXBuiltinSMAndPTX<"short(float, float)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_ff_to_e2m1x2_rn_relu_satfinite : NVPTXBuiltinSMAndPTX<"short(float, float)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;

def __nvvm_e2m1x2_to_f16x2_rn : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(short)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_e2m1x2_to_f16x2_rn_relu : NVPTXBuiltinSMAndPTX<"_Vector<2, __fp16>(short)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;

def __nvvm_f32x4_to_e2m1x4_rs_satfinite :
  NVPTXBuiltinSMAndPTX<"short(_Vector<4, float>, uint32_t)",
                       SM<"100a", [SM_103a]>, PTX87>;
def __nvvm_f32x4_to_e2m1x4_rs_relu_satfinite :
  NVPTXBuiltinSMAndPTX<"short(_Vector<4, float>, uint32_t)",
                       SM<"100a", [SM_103a]>, PTX87>;

def __nvvm_ff_to_ue8m0x2_rz : NVPTXBuiltinSMAndPTX<"short(float, float)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_ff_to_ue8m0x2_rz_satfinite : NVPTXBuiltinSMAndPTX<"short(float, float)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_ff_to_ue8m0x2_rp : NVPTXBuiltinSMAndPTX<"short(float, float)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_ff_to_ue8m0x2_rp_satfinite : NVPTXBuiltinSMAndPTX<"short(float, float)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;

def __nvvm_bf16x2_to_ue8m0x2_rz : NVPTXBuiltinSMAndPTX<"short(_Vector<2, __bf16>)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_bf16x2_to_ue8m0x2_rz_satfinite : NVPTXBuiltinSMAndPTX<"short(_Vector<2, __bf16>)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_bf16x2_to_ue8m0x2_rp : NVPTXBuiltinSMAndPTX<"short(_Vector<2, __bf16>)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;
def __nvvm_bf16x2_to_ue8m0x2_rp_satfinite : NVPTXBuiltinSMAndPTX<"short(_Vector<2, __bf16>)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;

def __nvvm_ue8m0x2_to_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(short)", SM<"100a", [SM_101a, SM_120a]>, PTX86>;

// FNS
let Attributes = [NoThrow] in {
  def __nvvm_fns : NVPTXBuiltinPTX<"unsigned int(unsigned int, unsigned int, int)", PTX60>;
}

// Sync

def __syncthreads : NVPTXBuiltin<"void()">;
def __nvvm_bar0_popc : NVPTXBuiltin<"int(int)">;
def __nvvm_bar0_and : NVPTXBuiltin<"int(int)">;
def __nvvm_bar0_or : NVPTXBuiltin<"int(int)">;
let Attributes = [NoThrow] in {
  def __nvvm_bar_sync : NVPTXBuiltin<"void(int)">;
  def __nvvm_bar_warp_sync : NVPTXBuiltinPTX<"void(unsigned int)", PTX60>;
  def __nvvm_barrier_sync : NVPTXBuiltinPTX<"void(unsigned int)", PTX60>;
  def __nvvm_barrier_sync_cnt : NVPTXBuiltinPTX<"void(unsigned int, unsigned int)", PTX60>;

  def __nvvm_barrier_cluster_arrive : NVPTXBuiltinSMAndPTX<"void()", SM_90, PTX78>;
  def __nvvm_barrier_cluster_arrive_relaxed : NVPTXBuiltinSMAndPTX<"void()", SM_90, PTX80>;
  def __nvvm_barrier_cluster_wait : NVPTXBuiltinSMAndPTX<"void()", SM_90, PTX78>;
  def __nvvm_fence_sc_cluster : NVPTXBuiltinSMAndPTX<"void()", SM_90, PTX78>;
}

// Shuffle

def __nvvm_shfl_down_i32 : NVPTXBuiltin<"int(int, int, int)">;
def __nvvm_shfl_down_f32 : NVPTXBuiltin<"float(float, int, int)">;
def __nvvm_shfl_up_i32 : NVPTXBuiltin<"int(int, int, int)">;
def __nvvm_shfl_up_f32 : NVPTXBuiltin<"float(float, int, int)">;
def __nvvm_shfl_bfly_i32 : NVPTXBuiltin<"int(int, int, int)">;
def __nvvm_shfl_bfly_f32 : NVPTXBuiltin<"float(float, int, int)">;
def __nvvm_shfl_idx_i32 : NVPTXBuiltin<"int(int, int, int)">;
def __nvvm_shfl_idx_f32 : NVPTXBuiltin<"float(float, int, int)">;

def __nvvm_shfl_sync_down_i32 : NVPTXBuiltinPTX<"int(unsigned int, int, int, int)", PTX60>;
def __nvvm_shfl_sync_down_f32 : NVPTXBuiltinPTX<"float(unsigned int, float, int, int)", PTX60>;
def __nvvm_shfl_sync_up_i32 : NVPTXBuiltinPTX<"int(unsigned int, int, int, int)", PTX60>;
def __nvvm_shfl_sync_up_f32 : NVPTXBuiltinPTX<"float(unsigned int, float, int, int)", PTX60>;
def __nvvm_shfl_sync_bfly_i32 : NVPTXBuiltinPTX<"int(unsigned int, int, int, int)", PTX60>;
def __nvvm_shfl_sync_bfly_f32 : NVPTXBuiltinPTX<"float(unsigned int, float, int, int)", PTX60>;
def __nvvm_shfl_sync_idx_i32 : NVPTXBuiltinPTX<"int(unsigned int, int, int, int)", PTX60>;
def __nvvm_shfl_sync_idx_f32 : NVPTXBuiltinPTX<"float(unsigned int, float, int, int)", PTX60>;

// Vote
def __nvvm_vote_all : NVPTXBuiltin<"bool(bool)">;
def __nvvm_vote_any : NVPTXBuiltin<"bool(bool)">;
def __nvvm_vote_uni : NVPTXBuiltin<"bool(bool)">;
def __nvvm_vote_ballot : NVPTXBuiltin<"unsigned int(bool)">;

def __nvvm_vote_all_sync : NVPTXBuiltinPTX<"bool(unsigned int, bool)", PTX60>;
def __nvvm_vote_any_sync : NVPTXBuiltinPTX<"bool(unsigned int, bool)", PTX60>;
def __nvvm_vote_uni_sync : NVPTXBuiltinPTX<"bool(unsigned int, bool)", PTX60>;
def __nvvm_vote_ballot_sync : NVPTXBuiltinPTX<"unsigned int(unsigned int, bool)", PTX60>;

// Mask
let Attributes = [NoThrow] in {
  def __nvvm_activemask : NVPTXBuiltinPTX<"unsigned int()", PTX62>;
}

// Match
def __nvvm_match_any_sync_i32 : NVPTXBuiltinSMAndPTX<"unsigned int(unsigned int, unsigned int)", SM_70, PTX60>;
def __nvvm_match_any_sync_i64 : NVPTXBuiltinSMAndPTX<"unsigned int(unsigned int, int64_t)", SM_70, PTX60>;
// These return a pair {value, predicate}, which requires custom lowering.
def __nvvm_match_all_sync_i32p : NVPTXBuiltinSMAndPTX<"unsigned int(unsigned int, unsigned int, int *)", SM_70, PTX60>;
def __nvvm_match_all_sync_i64p : NVPTXBuiltinSMAndPTX<"unsigned int(unsigned int, int64_t, int *)", SM_70, PTX60>;

// Redux
def __nvvm_redux_sync_add : NVPTXBuiltinSMAndPTX<"int(int, int)", SM_80, PTX70>;
def __nvvm_redux_sync_min : NVPTXBuiltinSMAndPTX<"int(int, int)", SM_80, PTX70>;
def __nvvm_redux_sync_max : NVPTXBuiltinSMAndPTX<"int(int, int)", SM_80, PTX70>;
def __nvvm_redux_sync_umin : NVPTXBuiltinSMAndPTX<"unsigned int(unsigned int, int)", SM_80, PTX70>;
def __nvvm_redux_sync_umax : NVPTXBuiltinSMAndPTX<"unsigned int(unsigned int, int)", SM_80, PTX70>;
def __nvvm_redux_sync_and : NVPTXBuiltinSMAndPTX<"int(int, int)", SM_80, PTX70>;
def __nvvm_redux_sync_xor : NVPTXBuiltinSMAndPTX<"int(int, int)", SM_80, PTX70>;
def __nvvm_redux_sync_or : NVPTXBuiltinSMAndPTX<"int(int, int)", SM_80, PTX70>;
def __nvvm_redux_sync_fmin : NVPTXBuiltinSMAndPTX<"float(float, int)", SM_100a, PTX86>;
def __nvvm_redux_sync_fmin_abs : NVPTXBuiltinSMAndPTX<"float(float, int)", SM_100a, PTX86>;
def __nvvm_redux_sync_fmin_NaN : NVPTXBuiltinSMAndPTX<"float(float, int)", SM_100a, PTX86>;
def __nvvm_redux_sync_fmin_abs_NaN : NVPTXBuiltinSMAndPTX<"float(float, int)", SM_100a, PTX86>;
def __nvvm_redux_sync_fmax : NVPTXBuiltinSMAndPTX<"float(float, int)", SM_100a, PTX86>;
def __nvvm_redux_sync_fmax_abs : NVPTXBuiltinSMAndPTX<"float(float, int)", SM_100a, PTX86>;
def __nvvm_redux_sync_fmax_NaN : NVPTXBuiltinSMAndPTX<"float(float, int)", SM_100a, PTX86>;
def __nvvm_redux_sync_fmax_abs_NaN : NVPTXBuiltinSMAndPTX<"float(float, int)", SM_100a, PTX86>;

// Membar

def __nvvm_membar_cta : NVPTXBuiltin<"void()">;
def __nvvm_membar_gl : NVPTXBuiltin<"void()">;
def __nvvm_membar_sys : NVPTXBuiltin<"void()">;

// mbarrier

def __nvvm_mbarrier_init : NVPTXBuiltinSMAndPTX<"void(int64_t *, int)", SM_80, PTX70>;
def __nvvm_mbarrier_init_shared : NVPTXBuiltinSMAndPTX<"void(int64_t address_space<3> *, int)", SM_80, PTX70>;

def __nvvm_mbarrier_inval : NVPTXBuiltinSMAndPTX<"void(int64_t *)", SM_80, PTX70>;
def __nvvm_mbarrier_inval_shared : NVPTXBuiltinSMAndPTX<"void(int64_t address_space<3> *)", SM_80, PTX70>;

def __nvvm_mbarrier_arrive : NVPTXBuiltinSMAndPTX<"int64_t(int64_t *)", SM_80, PTX70>;
def __nvvm_mbarrier_arrive_shared : NVPTXBuiltinSMAndPTX<"int64_t(int64_t address_space<3> *)", SM_80, PTX70>;
def __nvvm_mbarrier_arrive_noComplete : NVPTXBuiltinSMAndPTX<"int64_t(int64_t *, int)", SM_80, PTX70>;
def __nvvm_mbarrier_arrive_noComplete_shared : NVPTXBuiltinSMAndPTX<"int64_t(int64_t address_space<3> *, int)", SM_80, PTX70>;

def __nvvm_mbarrier_arrive_drop : NVPTXBuiltinSMAndPTX<"int64_t(int64_t *)", SM_80, PTX70>;
def __nvvm_mbarrier_arrive_drop_shared : NVPTXBuiltinSMAndPTX<"int64_t(int64_t address_space<3> *)", SM_80, PTX70>;
def __nvvm_mbarrier_arrive_drop_noComplete : NVPTXBuiltinSMAndPTX<"int64_t(int64_t *, int)", SM_80, PTX70>;
def __nvvm_mbarrier_arrive_drop_noComplete_shared : NVPTXBuiltinSMAndPTX<"int64_t(int64_t address_space<3> *, int)", SM_80, PTX70>;

def __nvvm_mbarrier_test_wait : NVPTXBuiltinSMAndPTX<"bool(int64_t *, int64_t)", SM_80, PTX70>;
def __nvvm_mbarrier_test_wait_shared : NVPTXBuiltinSMAndPTX<"bool(int64_t address_space<3> *, int64_t)", SM_80, PTX70>;

def __nvvm_mbarrier_pending_count : NVPTXBuiltinSMAndPTX<"int(int64_t)", SM_80, PTX70>;

// Memcpy, Memset

def __nvvm_memcpy : NVPTXBuiltin<"void(unsigned char *, unsigned char *, size_t, int)">;
def __nvvm_memset : NVPTXBuiltin<"void(unsigned char *, unsigned char, size_t, int)">;

// Image

def __builtin_ptx_read_image2Dfi_ : NVPTXBuiltin<"_Vector<4, float>(int, int, int, int)">;
def __builtin_ptx_read_image2Dff_ : NVPTXBuiltin<"_Vector<4, float>(int, int, float, float)">;
def __builtin_ptx_read_image2Dii_ : NVPTXBuiltin<"_Vector<4, int>(int, int, int, int)">;
def __builtin_ptx_read_image2Dif_ : NVPTXBuiltin<"_Vector<4, int>(int, int, float, float)">;

def __builtin_ptx_read_image3Dfi_ : NVPTXBuiltin<"_Vector<4, float>(int, int, int, int, int, int)">;
def __builtin_ptx_read_image3Dff_ : NVPTXBuiltin<"_Vector<4, float>(int, int, float, float, float, float)">;
def __builtin_ptx_read_image3Dii_ : NVPTXBuiltin<"_Vector<4, int>(int, int, int, int, int, int)">;
def __builtin_ptx_read_image3Dif_ : NVPTXBuiltin<"_Vector<4, int>(int, int, float, float, float, float)">;

def __builtin_ptx_write_image2Df_ : NVPTXBuiltin<"void(int, int, int, float, float, float, float)">;
def __builtin_ptx_write_image2Di_ : NVPTXBuiltin<"void(int, int, int, int, int, int, int)">;
def __builtin_ptx_write_image2Dui_ : NVPTXBuiltin<"void(int, int, int, unsigned int, unsigned int, unsigned int, unsigned int)">;
def __builtin_ptx_get_image_depthi_ : NVPTXBuiltin<"int(int)">;
def __builtin_ptx_get_image_heighti_ : NVPTXBuiltin<"int(int)">;
def __builtin_ptx_get_image_widthi_ : NVPTXBuiltin<"int(int)">;
def __builtin_ptx_get_image_channel_data_typei_ : NVPTXBuiltin<"int(int)">;
def __builtin_ptx_get_image_channel_orderi_ : NVPTXBuiltin<"int(int)">;

// Atomic
//
// We need the atom intrinsics because
// - they are used in converging analysis
// - they are used in address space analysis and optimization
// So it does not hurt to expose them as builtins.
//
let Attributes = [NoThrow] in {
  def __nvvm_atom_add_gen_i : NVPTXBuiltin<"int(int volatile *, int)">;
  def __nvvm_atom_cta_add_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_sys_add_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_add_gen_l : NVPTXBuiltin<"long int(long int volatile *, long int)">;
  def __nvvm_atom_cta_add_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_sys_add_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_add_gen_ll : NVPTXBuiltin<"long long int(long long int volatile *, long long int)">;
  def __nvvm_atom_cta_add_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;
  def __nvvm_atom_sys_add_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;
  def __nvvm_atom_add_gen_f : NVPTXBuiltin<"float(float volatile *, float)">;
  def __nvvm_atom_cta_add_gen_f : NVPTXBuiltinSM<"float(float volatile *, float)", SM_60>;
  def __nvvm_atom_sys_add_gen_f : NVPTXBuiltinSM<"float(float volatile *, float)", SM_60>;
  def __nvvm_atom_add_gen_d : NVPTXBuiltinSM<"double(double volatile *, double)", SM_60>;
  def __nvvm_atom_cta_add_gen_d : NVPTXBuiltinSM<"double(double volatile *, double)", SM_60>;
  def __nvvm_atom_sys_add_gen_d : NVPTXBuiltinSM<"double(double volatile *, double)", SM_60>;

  def __nvvm_atom_sub_gen_i : NVPTXBuiltin<"int(int volatile *, int)">;
  def __nvvm_atom_sub_gen_l : NVPTXBuiltin<"long int(long int volatile *, long int)">;
  def __nvvm_atom_sub_gen_ll : NVPTXBuiltin<"long long int(long long int volatile *, long long int)">;

  def __nvvm_atom_xchg_gen_i : NVPTXBuiltin<"int(int volatile *, int)">;
  def __nvvm_atom_cta_xchg_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_sys_xchg_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_xchg_gen_l : NVPTXBuiltin<"long int(long int volatile *, long int)">;
  def __nvvm_atom_cta_xchg_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_sys_xchg_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_xchg_gen_ll : NVPTXBuiltin<"long long int(long long int volatile *, long long int)">;
  def __nvvm_atom_cta_xchg_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;
  def __nvvm_atom_sys_xchg_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;

  def __nvvm_atom_max_gen_i : NVPTXBuiltin<"int(int volatile *, int)">;
  def __nvvm_atom_cta_max_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_sys_max_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_max_gen_ui : NVPTXBuiltin<"unsigned int(unsigned int volatile *, unsigned int)">;
  def __nvvm_atom_cta_max_gen_ui : NVPTXBuiltinSM<"unsigned int(unsigned int volatile *, unsigned int)", SM_60>;
  def __nvvm_atom_sys_max_gen_ui : NVPTXBuiltinSM<"unsigned int(unsigned int volatile *, unsigned int)", SM_60>;
  def __nvvm_atom_max_gen_l : NVPTXBuiltin<"long int(long int volatile *, long int)">;
  def __nvvm_atom_cta_max_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_sys_max_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_max_gen_ul : NVPTXBuiltin<"unsigned long int(unsigned long int volatile *, unsigned long int)">;
  def __nvvm_atom_cta_max_gen_ul : NVPTXBuiltinSM<"unsigned long int(unsigned long int volatile *, unsigned long int)", SM_60>;
  def __nvvm_atom_sys_max_gen_ul : NVPTXBuiltinSM<"unsigned long int(unsigned long int volatile *, unsigned long int)", SM_60>;
  def __nvvm_atom_max_gen_ll : NVPTXBuiltin<"long long int(long long int volatile *, long long int)">;
  def __nvvm_atom_cta_max_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;
  def __nvvm_atom_sys_max_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;
  def __nvvm_atom_max_gen_ull : NVPTXBuiltin<"unsigned long long int(unsigned long long int volatile *, unsigned long long int)">;
  def __nvvm_atom_cta_max_gen_ull : NVPTXBuiltinSM<"unsigned long long int(unsigned long long int volatile *, unsigned long long int)", SM_60>;
  def __nvvm_atom_sys_max_gen_ull : NVPTXBuiltinSM<"unsigned long long int(unsigned long long int volatile *, unsigned long long int)", SM_60>;

  def __nvvm_atom_min_gen_i : NVPTXBuiltin<"int(int volatile *, int)">;
  def __nvvm_atom_cta_min_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_sys_min_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_min_gen_ui : NVPTXBuiltin<"unsigned int(unsigned int volatile *, unsigned int)">;
  def __nvvm_atom_cta_min_gen_ui : NVPTXBuiltinSM<"unsigned int(unsigned int volatile *, unsigned int)", SM_60>;
  def __nvvm_atom_sys_min_gen_ui : NVPTXBuiltinSM<"unsigned int(unsigned int volatile *, unsigned int)", SM_60>;
  def __nvvm_atom_min_gen_l : NVPTXBuiltin<"long int(long int volatile *, long int)">;
  def __nvvm_atom_cta_min_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_sys_min_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_min_gen_ul : NVPTXBuiltin<"unsigned long int(unsigned long int volatile *, unsigned long int)">;
  def __nvvm_atom_cta_min_gen_ul : NVPTXBuiltinSM<"unsigned long int(unsigned long int volatile *, unsigned long int)", SM_60>;
  def __nvvm_atom_sys_min_gen_ul : NVPTXBuiltinSM<"unsigned long int(unsigned long int volatile *, unsigned long int)", SM_60>;
  def __nvvm_atom_min_gen_ll : NVPTXBuiltin<"long long int(long long int volatile *, long long int)">;
  def __nvvm_atom_cta_min_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;
  def __nvvm_atom_sys_min_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;
  def __nvvm_atom_min_gen_ull : NVPTXBuiltin<"unsigned long long int(unsigned long long int volatile *, unsigned long long int)">;
  def __nvvm_atom_cta_min_gen_ull : NVPTXBuiltinSM<"unsigned long long int(unsigned long long int volatile *, unsigned long long int)", SM_60>;
  def __nvvm_atom_sys_min_gen_ull : NVPTXBuiltinSM<"unsigned long long int(unsigned long long int volatile *, unsigned long long int)", SM_60>;

  def __nvvm_atom_inc_gen_ui : NVPTXBuiltin<"unsigned int(unsigned int volatile *, unsigned int)">;
  def __nvvm_atom_cta_inc_gen_ui : NVPTXBuiltinSM<"unsigned int(unsigned int volatile *, unsigned int)", SM_60>;
  def __nvvm_atom_sys_inc_gen_ui : NVPTXBuiltinSM<"unsigned int(unsigned int volatile *, unsigned int)", SM_60>;
  def __nvvm_atom_dec_gen_ui : NVPTXBuiltin<"unsigned int(unsigned int volatile *, unsigned int)">;
  def __nvvm_atom_cta_dec_gen_ui : NVPTXBuiltinSM<"unsigned int(unsigned int volatile *, unsigned int)", SM_60>;
  def __nvvm_atom_sys_dec_gen_ui : NVPTXBuiltinSM<"unsigned int(unsigned int volatile *, unsigned int)", SM_60>;

  def __nvvm_atom_and_gen_i : NVPTXBuiltin<"int(int volatile *, int)">;
  def __nvvm_atom_cta_and_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_sys_and_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_and_gen_l : NVPTXBuiltin<"long int(long int volatile *, long int)">;
  def __nvvm_atom_cta_and_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_sys_and_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_and_gen_ll : NVPTXBuiltin<"long long int(long long int volatile *, long long int)">;
  def __nvvm_atom_cta_and_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;
  def __nvvm_atom_sys_and_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;

  def __nvvm_atom_or_gen_i : NVPTXBuiltin<"int(int volatile *, int)">;
  def __nvvm_atom_cta_or_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_sys_or_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_or_gen_l : NVPTXBuiltin<"long int(long int volatile *, long int)">;
  def __nvvm_atom_cta_or_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_sys_or_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_or_gen_ll : NVPTXBuiltin<"long long int(long long int volatile *, long long int)">;
  def __nvvm_atom_cta_or_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;
  def __nvvm_atom_sys_or_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;

  def __nvvm_atom_xor_gen_i : NVPTXBuiltin<"int(int volatile *, int)">;
  def __nvvm_atom_cta_xor_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_sys_xor_gen_i : NVPTXBuiltinSM<"int(int volatile *, int)", SM_60>;
  def __nvvm_atom_xor_gen_l : NVPTXBuiltin<"long int(long int volatile *, long int)">;
  def __nvvm_atom_cta_xor_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_sys_xor_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int)", SM_60>;
  def __nvvm_atom_xor_gen_ll : NVPTXBuiltin<"long long int(long long int volatile *, long long int)">;
  def __nvvm_atom_cta_xor_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;
  def __nvvm_atom_sys_xor_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int)", SM_60>;

  def __nvvm_atom_cas_gen_us : NVPTXBuiltinSM<"unsigned short(unsigned short volatile *, unsigned short, unsigned short)", SM_70>;
  def __nvvm_atom_cta_cas_gen_us : NVPTXBuiltinSM<"unsigned short(unsigned short volatile *, unsigned short, unsigned short)", SM_70>;
  def __nvvm_atom_sys_cas_gen_us : NVPTXBuiltinSM<"unsigned short(unsigned short volatile *, unsigned short, unsigned short)", SM_70>;
  def __nvvm_atom_cas_gen_i : NVPTXBuiltin<"int(int volatile *, int, int)">;
  def __nvvm_atom_cta_cas_gen_i : NVPTXBuiltinSM<"int(int volatile *, int, int)", SM_60>;
  def __nvvm_atom_sys_cas_gen_i : NVPTXBuiltinSM<"int(int volatile *, int, int)", SM_60>;
  def __nvvm_atom_cas_gen_l : NVPTXBuiltin<"long int(long int volatile *, long int, long int)">;
  def __nvvm_atom_cta_cas_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int, long int)", SM_60>;
  def __nvvm_atom_sys_cas_gen_l : NVPTXBuiltinSM<"long int(long int volatile *, long int, long int)", SM_60>;
  def __nvvm_atom_cas_gen_ll : NVPTXBuiltin<"long long int(long long int volatile *, long long int, long long int)">;
  def __nvvm_atom_cta_cas_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int, long long int)", SM_60>;
  def __nvvm_atom_sys_cas_gen_ll : NVPTXBuiltinSM<"long long int(long long int volatile *, long long int, long long int)", SM_60>;
}

// Compiler Error Warn
let Attributes = [NoThrow] in {
  def __nvvm_compiler_error : NVPTXBuiltin<"void(char const address_space<4> *)">;
  def __nvvm_compiler_warn : NVPTXBuiltin<"void(char const address_space<4> *)">;
}

def __nvvm_ldu_c : NVPTXBuiltin<"char(char const *)">;
def __nvvm_ldu_sc : NVPTXBuiltin<"signed char(signed char const *)">;
def __nvvm_ldu_s : NVPTXBuiltin<"short(short const *)">;
def __nvvm_ldu_i : NVPTXBuiltin<"int(int const *)">;
def __nvvm_ldu_l : NVPTXBuiltin<"long int(long int const *)">;
def __nvvm_ldu_ll : NVPTXBuiltin<"long long int(long long int const *)">;

def __nvvm_ldu_uc : NVPTXBuiltin<"unsigned char(unsigned char const *)">;
def __nvvm_ldu_us : NVPTXBuiltin<"unsigned short(unsigned short const *)">;
def __nvvm_ldu_ui : NVPTXBuiltin<"unsigned int(unsigned int const *)">;
def __nvvm_ldu_ul : NVPTXBuiltin<"unsigned long int(unsigned long int const *)">;
def __nvvm_ldu_ull : NVPTXBuiltin<"unsigned long long int(unsigned long long int const *)">;

def __nvvm_ldu_h : NVPTXBuiltin<"__fp16(__fp16 const *)">;
def __nvvm_ldu_f : NVPTXBuiltin<"float(float const *)">;
def __nvvm_ldu_d : NVPTXBuiltin<"double(double const *)">;

def __nvvm_ldu_c2 : NVPTXBuiltin<"_ExtVector<2, char>(_ExtVector<2, char const *>)">;
def __nvvm_ldu_sc2 : NVPTXBuiltin<"_ExtVector<2, signed char>(_ExtVector<2, signed char const *>)">;
def __nvvm_ldu_c4 : NVPTXBuiltin<"_ExtVector<4, char>(_ExtVector<4, char const *>)">;
def __nvvm_ldu_sc4 : NVPTXBuiltin<"_ExtVector<4, signed char>(_ExtVector<4, signed char const *>)">;
def __nvvm_ldu_s2 : NVPTXBuiltin<"_ExtVector<2, short>(_ExtVector<2, short const *>)">;
def __nvvm_ldu_s4 : NVPTXBuiltin<"_ExtVector<4, short>(_ExtVector<4, short const *>)">;
def __nvvm_ldu_i2 : NVPTXBuiltin<"_ExtVector<2, int>(_ExtVector<2, int const *>)">;
def __nvvm_ldu_i4 : NVPTXBuiltin<"_ExtVector<4, int>(_ExtVector<4, int const *>)">;
def __nvvm_ldu_l2 : NVPTXBuiltin<"_ExtVector<2, long int>(_ExtVector<2, long int const *>)">;
def __nvvm_ldu_ll2 : NVPTXBuiltin<"_ExtVector<2, long long int>(_ExtVector<2, long long int const *>)">;

def __nvvm_ldu_uc2 : NVPTXBuiltin<"_ExtVector<2, unsigned char>(_ExtVector<2, unsigned char const *>)">;
def __nvvm_ldu_uc4 : NVPTXBuiltin<"_ExtVector<4, unsigned char>(_ExtVector<4, unsigned char const *>)">;
def __nvvm_ldu_us2 : NVPTXBuiltin<"_ExtVector<2, unsigned short>(_ExtVector<2, unsigned short const *>)">;
def __nvvm_ldu_us4 : NVPTXBuiltin<"_ExtVector<4, unsigned short>(_ExtVector<4, unsigned short const *>)">;
def __nvvm_ldu_ui2 : NVPTXBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<2, unsigned int const *>)">;
def __nvvm_ldu_ui4 : NVPTXBuiltin<"_ExtVector<4, unsigned int>(_ExtVector<4, unsigned int const *>)">;
def __nvvm_ldu_ul2 : NVPTXBuiltin<"_ExtVector<2, unsigned long int>(_ExtVector<2, unsigned long int const *>)">;
def __nvvm_ldu_ull2 : NVPTXBuiltin<"_ExtVector<2, unsigned long long int>(_ExtVector<2, unsigned long long int const *>)">;

def __nvvm_ldu_h2 : NVPTXBuiltin<"_ExtVector<2, __fp16>(_ExtVector<2, __fp16 const *>)">;
def __nvvm_ldu_f2 : NVPTXBuiltin<"_ExtVector<2, float>(_ExtVector<2, float const *>)">;
def __nvvm_ldu_f4 : NVPTXBuiltin<"_ExtVector<4, float>(_ExtVector<4, float const *>)">;
def __nvvm_ldu_d2 : NVPTXBuiltin<"_ExtVector<2, double>(_ExtVector<2, double const *>)">;

def __nvvm_ldg_c : NVPTXBuiltin<"char(char const *)">;
def __nvvm_ldg_sc : NVPTXBuiltin<"signed char(signed char const *)">;
def __nvvm_ldg_s : NVPTXBuiltin<"short(short const *)">;
def __nvvm_ldg_i : NVPTXBuiltin<"int(int const *)">;
def __nvvm_ldg_l : NVPTXBuiltin<"long int(long int const *)">;
def __nvvm_ldg_ll : NVPTXBuiltin<"long long int(long long int const *)">;

def __nvvm_ldg_uc : NVPTXBuiltin<"unsigned char(unsigned char const *)">;
def __nvvm_ldg_us : NVPTXBuiltin<"unsigned short(unsigned short const *)">;
def __nvvm_ldg_ui : NVPTXBuiltin<"unsigned int(unsigned int const *)">;
def __nvvm_ldg_ul : NVPTXBuiltin<"unsigned long int(unsigned long int const *)">;
def __nvvm_ldg_ull : NVPTXBuiltin<"unsigned long long int(unsigned long long int const *)">;

def __nvvm_ldg_h : NVPTXBuiltin<"__fp16(__fp16 const *)">;
def __nvvm_ldg_f : NVPTXBuiltin<"float(float const *)">;
def __nvvm_ldg_d : NVPTXBuiltin<"double(double const *)">;

def __nvvm_ldg_c2 : NVPTXBuiltin<"_ExtVector<2, char>(_ExtVector<2, char const *>)">;
def __nvvm_ldg_sc2 : NVPTXBuiltin<"_ExtVector<2, signed char>(_ExtVector<2, signed char const *>)">;
def __nvvm_ldg_c4 : NVPTXBuiltin<"_ExtVector<4, char>(_ExtVector<4, char const *>)">;
def __nvvm_ldg_sc4 : NVPTXBuiltin<"_ExtVector<4, signed char>(_ExtVector<4, signed char const *>)">;
def __nvvm_ldg_s2 : NVPTXBuiltin<"_ExtVector<2, short>(_ExtVector<2, short const *>)">;
def __nvvm_ldg_s4 : NVPTXBuiltin<"_ExtVector<4, short>(_ExtVector<4, short const *>)">;
def __nvvm_ldg_i2 : NVPTXBuiltin<"_ExtVector<2, int>(_ExtVector<2, int const *>)">;
def __nvvm_ldg_i4 : NVPTXBuiltin<"_ExtVector<4, int>(_ExtVector<4, int const *>)">;
def __nvvm_ldg_l2 : NVPTXBuiltin<"_ExtVector<2, long int>(_ExtVector<2, long int const *>)">;
def __nvvm_ldg_ll2 : NVPTXBuiltin<"_ExtVector<2, long long int>(_ExtVector<2, long long int const *>)">;

def __nvvm_ldg_uc2 : NVPTXBuiltin<"_ExtVector<2, unsigned char>(_ExtVector<2, unsigned char const *>)">;
def __nvvm_ldg_uc4 : NVPTXBuiltin<"_ExtVector<4, unsigned char>(_ExtVector<4, unsigned char const *>)">;
def __nvvm_ldg_us2 : NVPTXBuiltin<"_ExtVector<2, unsigned short>(_ExtVector<2, unsigned short const *>)">;
def __nvvm_ldg_us4 : NVPTXBuiltin<"_ExtVector<4, unsigned short>(_ExtVector<4, unsigned short const *>)">;
def __nvvm_ldg_ui2 : NVPTXBuiltin<"_ExtVector<2, unsigned int>(_ExtVector<2, unsigned int const *>)">;
def __nvvm_ldg_ui4 : NVPTXBuiltin<"_ExtVector<4, unsigned int>(_ExtVector<4, unsigned int const *>)">;
def __nvvm_ldg_ul2 : NVPTXBuiltin<"_ExtVector<2, unsigned long int>(_ExtVector<2, unsigned long int const *>)">;
def __nvvm_ldg_ull2 : NVPTXBuiltin<"_ExtVector<2, unsigned long long int>(_ExtVector<2, unsigned long long int const *>)">;

def __nvvm_ldg_h2 : NVPTXBuiltin<"_ExtVector<2, __fp16>(_ExtVector<2, __fp16 const *>)">;
def __nvvm_ldg_f2 : NVPTXBuiltin<"_ExtVector<2, float>(_ExtVector<2, float const *>)">;
def __nvvm_ldg_f4 : NVPTXBuiltin<"_ExtVector<4, float>(_ExtVector<4, float const *>)">;
def __nvvm_ldg_d2 : NVPTXBuiltin<"_ExtVector<2, double>(_ExtVector<2, double const *>)">;

// Address space predicates.
let Attributes = [NoThrow, Const] in {
  def __nvvm_isspacep_const : NVPTXBuiltin<"bool(void const *)">;
  def __nvvm_isspacep_global : NVPTXBuiltin<"bool(void const *)">;
  def __nvvm_isspacep_local : NVPTXBuiltin<"bool(void const *)">;
  def __nvvm_isspacep_shared : NVPTXBuiltin<"bool(void const *)">;
  def __nvvm_isspacep_shared_cluster : NVPTXBuiltinSMAndPTX<"bool(void const *)", SM_90, PTX78>;
}

// Builtins to support WMMA instructions on sm_70
def __hmma_m16n16k16_ld_a : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX60>;
def __hmma_m16n16k16_ld_b : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX60>;
def __hmma_m16n16k16_ld_c_f16 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX60>;
def __hmma_m16n16k16_ld_c_f32 : NVPTXBuiltinSMAndPTX<"void(float *, float const *, unsigned int, _Constant int)", SM_70, PTX60>;
def __hmma_m16n16k16_st_c_f16 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX60>;
def __hmma_m16n16k16_st_c_f32 : NVPTXBuiltinSMAndPTX<"void(float *, float const *, unsigned int, _Constant int)", SM_70, PTX60>;

def __hmma_m32n8k16_ld_a : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX61>;
def __hmma_m32n8k16_ld_b : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX61>;
def __hmma_m32n8k16_ld_c_f16 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX61>;
def __hmma_m32n8k16_ld_c_f32 : NVPTXBuiltinSMAndPTX<"void(float *, float const *, unsigned int, _Constant int)", SM_70, PTX61>;
def __hmma_m32n8k16_st_c_f16 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX61>;
def __hmma_m32n8k16_st_c_f32 : NVPTXBuiltinSMAndPTX<"void(float *, float const *, unsigned int, _Constant int)", SM_70, PTX61>;

def __hmma_m8n32k16_ld_a : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX61>;
def __hmma_m8n32k16_ld_b : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX61>;
def __hmma_m8n32k16_ld_c_f16 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX61>;
def __hmma_m8n32k16_ld_c_f32 : NVPTXBuiltinSMAndPTX<"void(float *, float const *, unsigned int, _Constant int)", SM_70, PTX61>;
def __hmma_m8n32k16_st_c_f16 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_70, PTX61>;
def __hmma_m8n32k16_st_c_f32 : NVPTXBuiltinSMAndPTX<"void(float *, float const *, unsigned int, _Constant int)", SM_70, PTX61>;

def __hmma_m16n16k16_mma_f16f16 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_70, PTX60>;
def __hmma_m16n16k16_mma_f32f16 : NVPTXBuiltinSMAndPTX<"void(float *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_70, PTX60>;
def __hmma_m16n16k16_mma_f32f32 : NVPTXBuiltinSMAndPTX<"void(float *, int const *, int const *, float const *, _Constant int, _Constant int)", SM_70, PTX60>;
def __hmma_m16n16k16_mma_f16f32 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, float const *, _Constant int, _Constant int)", SM_70, PTX60>;

def __hmma_m32n8k16_mma_f16f16 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_70, PTX61>;
def __hmma_m32n8k16_mma_f32f16 : NVPTXBuiltinSMAndPTX<"void(float *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_70, PTX61>;
def __hmma_m32n8k16_mma_f32f32 : NVPTXBuiltinSMAndPTX<"void(float *, int const *, int const *, float const *, _Constant int, _Constant int)", SM_70, PTX61>;
def __hmma_m32n8k16_mma_f16f32 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, float const *, _Constant int, _Constant int)", SM_70, PTX61>;

def __hmma_m8n32k16_mma_f16f16 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_70, PTX61>;
def __hmma_m8n32k16_mma_f32f16 : NVPTXBuiltinSMAndPTX<"void(float *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_70, PTX61>;
def __hmma_m8n32k16_mma_f32f32 : NVPTXBuiltinSMAndPTX<"void(float *, int const *, int const *, float const *, _Constant int, _Constant int)", SM_70, PTX61>;
def __hmma_m8n32k16_mma_f16f32 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, float const *, _Constant int, _Constant int)", SM_70, PTX61>;

// Builtins to support integer and sub-integer WMMA instructions on sm_72/sm_75
def __bmma_m8n8k128_ld_a_b1 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_75, PTX63>;
def __bmma_m8n8k128_ld_b_b1 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_75, PTX63>;
def __bmma_m8n8k128_ld_c : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_75, PTX63>;
def __bmma_m8n8k128_mma_and_popc_b1 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int)", SM_80, PTX71>;
def __bmma_m8n8k128_mma_xor_popc_b1 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int)", SM_75, PTX63>;
def __bmma_m8n8k128_st_c_i32 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_75, PTX63>;
def __imma_m16n16k16_ld_a_s8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m16n16k16_ld_a_u8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m16n16k16_ld_b_s8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m16n16k16_ld_b_u8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m16n16k16_ld_c : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m16n16k16_mma_s8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_72, PTX63>;
def __imma_m16n16k16_mma_u8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_72, PTX63>;
def __imma_m16n16k16_st_c_i32 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m32n8k16_ld_a_s8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m32n8k16_ld_a_u8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m32n8k16_ld_b_s8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m32n8k16_ld_b_u8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m32n8k16_ld_c : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m32n8k16_mma_s8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_72, PTX63>;
def __imma_m32n8k16_mma_u8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_72, PTX63>;
def __imma_m32n8k16_st_c_i32 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m8n32k16_ld_a_s8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m8n32k16_ld_a_u8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m8n32k16_ld_b_s8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m8n32k16_ld_b_u8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m8n32k16_ld_c : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m8n32k16_mma_s8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_72, PTX63>;
def __imma_m8n32k16_mma_u8 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_72, PTX63>;
def __imma_m8n32k16_st_c_i32 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_72, PTX63>;
def __imma_m8n8k32_ld_a_s4 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_75, PTX63>;
def __imma_m8n8k32_ld_a_u4 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_75, PTX63>;
def __imma_m8n8k32_ld_b_s4 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_75, PTX63>;
def __imma_m8n8k32_ld_b_u4 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_75, PTX63>;
def __imma_m8n8k32_ld_c : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_75, PTX63>;
def __imma_m8n8k32_mma_s4 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_75, PTX63>;
def __imma_m8n8k32_mma_u4 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, int const *, int const *, _Constant int, _Constant int)", SM_75, PTX63>;
def __imma_m8n8k32_st_c_i32 : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_75, PTX63>;

// Builtins to support double and alternate float WMMA instructions on sm_80
def __dmma_m8n8k4_ld_a : NVPTXBuiltinSMAndPTX<"void(double *, double const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __dmma_m8n8k4_ld_b : NVPTXBuiltinSMAndPTX<"void(double *, double const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __dmma_m8n8k4_ld_c : NVPTXBuiltinSMAndPTX<"void(double *, double const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __dmma_m8n8k4_st_c_f64 : NVPTXBuiltinSMAndPTX<"void(double *, double const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __dmma_m8n8k4_mma_f64 : NVPTXBuiltinSMAndPTX<"void(double *, double const *, double const *, double const *, _Constant int, _Constant int)", SM_80, PTX70>;

def __mma_bf16_m16n16k16_ld_a : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __mma_bf16_m16n16k16_ld_b : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __mma_bf16_m16n16k16_mma_f32 : NVPTXBuiltinSMAndPTX<"void(float *, int const *, int const *, float const *, _Constant int, _Constant int)", SM_80, PTX70>;
def __mma_bf16_m8n32k16_ld_a : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __mma_bf16_m8n32k16_ld_b : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __mma_bf16_m8n32k16_mma_f32 : NVPTXBuiltinSMAndPTX<"void(float *, int const *, int const *, float const *, _Constant int, _Constant int)", SM_80, PTX70>;
def __mma_bf16_m32n8k16_ld_a : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __mma_bf16_m32n8k16_ld_b : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __mma_bf16_m32n8k16_mma_f32 : NVPTXBuiltinSMAndPTX<"void(float *, int const *, int const *, float const *, _Constant int, _Constant int)", SM_80, PTX70>;

def __mma_tf32_m16n16k8_ld_a : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __mma_tf32_m16n16k8_ld_b : NVPTXBuiltinSMAndPTX<"void(int *, int const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __mma_tf32_m16n16k8_ld_c : NVPTXBuiltinSMAndPTX<"void(float *, float const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __mma_m16n16k8_st_c_f32 : NVPTXBuiltinSMAndPTX<"void(float *, float const *, unsigned int, _Constant int)", SM_80, PTX70>;
def __mma_tf32_m16n16k8_mma_f32 : NVPTXBuiltinSMAndPTX<"void(float *, int const *, int const *, float const *, _Constant int, _Constant int)", SM_80, PTX70>;

// Async Copy
def __nvvm_cp_async_mbarrier_arrive : NVPTXBuiltinSMAndPTX<"void(int64_t *)", SM_80, PTX70>;
def __nvvm_cp_async_mbarrier_arrive_shared : NVPTXBuiltinSMAndPTX<"void(int64_t address_space<3> *)", SM_80, PTX70>;
def __nvvm_cp_async_mbarrier_arrive_noinc : NVPTXBuiltinSMAndPTX<"void(int64_t *)", SM_80, PTX70>;
def __nvvm_cp_async_mbarrier_arrive_noinc_shared : NVPTXBuiltinSMAndPTX<"void(int64_t address_space<3> *)", SM_80, PTX70>;

def __nvvm_cp_async_ca_shared_global_4 : NVPTXBuiltinSMAndPTX<"void(void address_space<3> *, void const address_space<1> *, ...)", SM_80, PTX70>;
def __nvvm_cp_async_ca_shared_global_8 : NVPTXBuiltinSMAndPTX<"void(void address_space<3> *, void const address_space<1> *, ...)", SM_80, PTX70>;
def __nvvm_cp_async_ca_shared_global_16 : NVPTXBuiltinSMAndPTX<"void(void address_space<3> *, void const address_space<1> *, ...)", SM_80, PTX70>;
def __nvvm_cp_async_cg_shared_global_16 : NVPTXBuiltinSMAndPTX<"void(void address_space<3> *, void const address_space<1> *, ...)", SM_80, PTX70>;

def __nvvm_cp_async_commit_group : NVPTXBuiltinSMAndPTX<"void()", SM_80, PTX70>;
def __nvvm_cp_async_wait_group : NVPTXBuiltinSMAndPTX<"void(_Constant int)", SM_80, PTX70>;
def __nvvm_cp_async_wait_all : NVPTXBuiltinSMAndPTX<"void()", SM_80, PTX70>;


// bf16, bf16x2 abs, neg
def __nvvm_abs_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16)", SM_80, PTX70>;
def __nvvm_abs_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>)", SM_80, PTX70>;
def __nvvm_neg_bf16 : NVPTXBuiltinSMAndPTX<"__bf16(__bf16)", SM_80, PTX70>;
def __nvvm_neg_bf16x2 : NVPTXBuiltinSMAndPTX<"_Vector<2, __bf16>(_Vector<2, __bf16>)", SM_80, PTX70>;

def __nvvm_mapa : NVPTXBuiltinSMAndPTX<"void *(void *, int)", SM_90, PTX78>;
def __nvvm_mapa_shared_cluster : NVPTXBuiltinSMAndPTX<"void address_space<3> *(void address_space<3> *, int)", SM_90, PTX78>;
def __nvvm_getctarank : NVPTXBuiltinSMAndPTX<"int(void *)", SM_90, PTX78>;
def __nvvm_getctarank_shared_cluster : NVPTXBuiltinSMAndPTX<"int(void address_space<3> *)", SM_90, PTX78>;
