//==--- riscv_sifive_vector.td - RISC-V SiFive VCIX function list ---------===//
//
//  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the builtins for RISC-V SiFive VCIX. See:
//
//     https://sifive.cdn.prismic.io/sifive/c3829e36-8552-41f0-a841-79945784241b_vcix-spec-software.pdf
//
//===----------------------------------------------------------------------===//

include "riscv_vector_common.td"

class IsFloat<string type> {
  bit val = !or(!eq(type, "x"), !eq(type, "f"), !eq(type, "d"), !eq(type, "y"));
}

//===----------------------------------------------------------------------===//
// Instruction definitions
//===----------------------------------------------------------------------===//

class VCIXSuffix<string range> {
  list<string> suffix = !cond(!eq(range, "c"): ["8mf8", "8mf4", "8mf2", "8m1", "8m2", "8m4", "8m8"],
                              !eq(range, "s"): ["16mf4", "16mf2", "16m1", "16m2", "16m4", "16m8"],
                              !eq(range, "i"): ["32mf2", "32m1", "32m2", "32m4", "32m8"],
                              !eq(range, "l"): ["64m1", "64m2", "64m4", "64m8"]);
}

class VCIXBuiltinSet<string name, string IR_name, string suffix,
                     string prototype, string type_range,
                     list<int> intrinsic_types>
    : RVVBuiltin<suffix, prototype, type_range> {
  let Name = name;
  let OverloadedName = name;
  let IRName = IR_name;
  let HasMasked = false;
  let IntrinsicTypes = intrinsic_types;
}

multiclass VCIXBuiltinSet<string name, string IR_name, string suffix,
                          string prototype, string type_range,
                          list<int> intrinsic_types> {
  if !find(prototype, "0") then {
    def : VCIXBuiltinSet<name, IR_name, suffix, prototype, type_range, intrinsic_types>;
  }
  def : VCIXBuiltinSet<name # "_se", IR_name # "_se", suffix, prototype, type_range, intrinsic_types>;
}

multiclass RVVVCIXBuiltinSet<list<string> range, string prototype,
                             list<int> intrinsic_types, bit UseGPR,
                             string suffix = "Uv"> {
  foreach r = range in
    let RequiredFeatures = !if(!and(UseGPR, !eq(r, "l")),
                               ["xsfvcp", "64bit"], ["xsfvcp"]) in
      defm : VCIXBuiltinSet<NAME, NAME, suffix, prototype, r, intrinsic_types>;
}

multiclass RVVVCIXBuiltinSetWOSuffix<list<string> range, string prototype,
                                     list<int> intrinsic_types, bit UseGPR> {
  let Log2LMUL = [0] in
  defm NAME : RVVVCIXBuiltinSet<range, prototype, intrinsic_types, UseGPR, "">;
}

let SupportOverloading = false in {
  defm sf_vc_x  : RVVVCIXBuiltinSetWOSuffix<["c", "s", "i", "l"], "0KzKzKzUeKzKz", [0, 3], UseGPR=1>;
  defm sf_vc_i  : RVVVCIXBuiltinSetWOSuffix<["i"], "0KzKzKzKzKzKz", [2, 3], UseGPR=0>;
  defm sf_vc_xv    : RVVVCIXBuiltinSet<["csi", "l"], "0KzKzUvUe",  [0, 2, 3],  UseGPR=1>;
  defm sf_vc_iv    : RVVVCIXBuiltinSet<["csi", "l"], "0KzKzUvKz",  [0, 2, 3],  UseGPR=0>;
  defm sf_vc_vv    : RVVVCIXBuiltinSet<["csi", "l"], "0KzKzUvUv",  [0, 2, 3],  UseGPR=0>;
  defm sf_vc_fv    : RVVVCIXBuiltinSet<["si",  "l"], "0KzKzUvFe",  [0, 2, 3],  UseGPR=0>;
  defm sf_vc_xvv   : RVVVCIXBuiltinSet<["csi", "l"], "0KzUvUvUe",  [0, 1, 2, 3],  UseGPR=1>;
  defm sf_vc_ivv   : RVVVCIXBuiltinSet<["csi", "l"], "0KzUvUvKz",  [0, 1, 2, 3],  UseGPR=0>;
  defm sf_vc_vvv   : RVVVCIXBuiltinSet<["csi", "l"], "0KzUvUvUv",  [0, 1, 2, 3],  UseGPR=0>;
  defm sf_vc_fvv   : RVVVCIXBuiltinSet<["si",  "l"], "0KzUvUvFe",  [0, 1, 2, 3],  UseGPR=0>;
  defm sf_vc_v_x   : RVVVCIXBuiltinSet<["csi", "l"], "UvKzKzUe",   [-1, 1, 2], UseGPR=1>;
  defm sf_vc_v_i   : RVVVCIXBuiltinSet<["csi", "l"], "UvKzKzKz",   [-1, 1, 2], UseGPR=0>;
  defm sf_vc_v_xv  : RVVVCIXBuiltinSet<["csi", "l"], "UvKzUvUe",   [-1, 0, 1, 2], UseGPR=1>;
  defm sf_vc_v_iv  : RVVVCIXBuiltinSet<["csi", "l"], "UvKzUvKz",   [-1, 0, 1, 2], UseGPR=0>;
  defm sf_vc_v_vv  : RVVVCIXBuiltinSet<["csi", "l"], "UvKzUvUv",   [-1, 0, 1, 2], UseGPR=0>;
  defm sf_vc_v_fv  : RVVVCIXBuiltinSet<["si",  "l"], "UvKzUvFe",   [-1, 0, 1, 2], UseGPR=0>;
  defm sf_vc_v_xvv : RVVVCIXBuiltinSet<["csi", "l"], "UvKzUvUvUe", [-1, 0, 1, 2, 3], UseGPR=1>;
  defm sf_vc_v_ivv : RVVVCIXBuiltinSet<["csi", "l"], "UvKzUvUvKz", [-1, 0, 1, 2, 3], UseGPR=0>;
  defm sf_vc_v_vvv : RVVVCIXBuiltinSet<["csi", "l"], "UvKzUvUvUv", [-1, 0, 1, 2, 3], UseGPR=0>;
  defm sf_vc_v_fvv : RVVVCIXBuiltinSet<["si",  "l"], "UvKzUvUvFe", [-1, 0, 1, 2, 3], UseGPR=0>;
  let Log2LMUL = [-3, -2, -1, 0, 1, 2] in {
    defm sf_vc_xvw   : RVVVCIXBuiltinSet<["csi"], "0KzUwUvUe",  [0, 1, 2, 3],  UseGPR=1>;
    defm sf_vc_ivw   : RVVVCIXBuiltinSet<["csi"], "0KzUwUvKz",  [0, 1, 2, 3],  UseGPR=0>;
    defm sf_vc_vvw   : RVVVCIXBuiltinSet<["csi"], "0KzUwUvUv",  [0, 1, 2, 3],  UseGPR=0>;
    defm sf_vc_fvw   : RVVVCIXBuiltinSet<["si"],  "0KzUwUvFe",  [0, 1, 2, 3],  UseGPR=0>;
    defm sf_vc_v_xvw : RVVVCIXBuiltinSet<["csi"], "UwKzUwUvUe", [-1, 0, 1, 2, 3], UseGPR=1>;
    defm sf_vc_v_ivw : RVVVCIXBuiltinSet<["csi"], "UwKzUwUvKz", [-1, 0, 1, 2, 3], UseGPR=0>;
    defm sf_vc_v_vvw : RVVVCIXBuiltinSet<["csi"], "UwKzUwUvUv", [-1, 0, 1, 2, 3], UseGPR=0>;
    defm sf_vc_v_fvw : RVVVCIXBuiltinSet<["si"],  "UwKzUwUvFe", [-1, 0, 1, 2, 3], UseGPR=0>;
  }
}

multiclass RVVVFWMACCBuiltinSet<list<list<string>> suffixes_prototypes> {
  let OverloadedName = NAME,
      Name = NAME,
      HasMasked = false,
      Log2LMUL = [-2, -1, 0, 1, 2] in
    defm NAME : RVVOutOp1Op2BuiltinSet<NAME, "y", suffixes_prototypes>;
}

multiclass RVVVQMACCDODBuiltinSet<list<list<string>> suffixes_prototypes> {
  let OverloadedName = NAME,
      Name = NAME,
      HasMasked = false,
      Log2LMUL = [0, 1, 2, 3] in
    defm NAME : RVVOutOp1Op2BuiltinSet<NAME, "i", suffixes_prototypes>;
}

multiclass RVVVQMACCQOQBuiltinSet<list<list<string>> suffixes_prototypes> {
  let OverloadedName = NAME,
      Name = NAME,
      HasMasked = false,
      Log2LMUL = [-1, 0, 1, 2] in
    defm NAME : RVVOutOp1Op2BuiltinSet<NAME, "s", suffixes_prototypes>;
}

multiclass RVVVFEXPBuiltinSet<list<list<string>> suffixes_prototypes, string type_range> {
  let UnMaskedPolicyScheme = HasPassthruOperand,
      OverloadedName = NAME,
      Log2LMUL = [-2, -1, 0, 1, 2, 3] in
    defm NAME : RVVOutBuiltinSet<NAME, type_range, suffixes_prototypes>;
}

multiclass RVVVFNRCLIPBuiltinSet<string suffix, string prototype, string type_range> {
  let Log2LMUL = [-3, -2, -1, 0, 1, 2],
      Name = NAME,
      IRName = NAME,
      MaskedIRName = NAME # "_mask" in
  def : RVVConvBuiltin<suffix, prototype, type_range, NAME>;
}

let UnMaskedPolicyScheme = HasPolicyOperand in
  let RequiredFeatures = ["xsfvqmaccdod"] in {
    defm sf_vqmaccu_2x8x2 : RVVVQMACCDODBuiltinSet<[["", "v", "vv(FixedSEW:8)SUv(FixedSEW:8)Uv"]]>;
    defm sf_vqmacc_2x8x2 : RVVVQMACCDODBuiltinSet<[["", "v", "vv(FixedSEW:8)Sv(FixedSEW:8)v"]]>;
    defm sf_vqmaccus_2x8x2 : RVVVQMACCDODBuiltinSet<[["", "v", "vv(FixedSEW:8)SUv(FixedSEW:8)v"]]>;
    defm sf_vqmaccsu_2x8x2 : RVVVQMACCDODBuiltinSet<[["", "v", "vv(FixedSEW:8)Sv(FixedSEW:8)Uv"]]>;
  }

let UnMaskedPolicyScheme = HasPolicyOperand in
  let RequiredFeatures = ["xsfvqmaccqoq"] in {
    defm sf_vqmaccu_4x8x4 : RVVVQMACCQOQBuiltinSet<[["", "w", "ww(FixedSEW:8)SUv(FixedSEW:8)Uv"]]>;
    defm sf_vqmacc_4x8x4 : RVVVQMACCQOQBuiltinSet<[["", "w", "ww(FixedSEW:8)Sv(FixedSEW:8)v"]]>;
    defm sf_vqmaccus_4x8x4 : RVVVQMACCQOQBuiltinSet<[["", "w", "ww(FixedSEW:8)SUv(FixedSEW:8)v"]]>;
    defm sf_vqmaccsu_4x8x4 : RVVVQMACCQOQBuiltinSet<[["", "w", "ww(FixedSEW:8)Sv(FixedSEW:8)Uv"]]>;
  }

let RequiredFeatures = ["xsfvfbfexp16e"] in {
  defm sf_vfexp : RVVVFEXPBuiltinSet<[["v", "v", "vv"]], "y">;
}

let RequiredFeatures = ["xsfvfexp16e"] in {
  defm sf_vfexp : RVVVFEXPBuiltinSet<[["v", "v", "vv"]], "x">;
}

let RequiredFeatures = ["xsfvfexp32e"] in {
  defm sf_vfexp : RVVVFEXPBuiltinSet<[["v", "v", "vv"]], "f">;
}

let RequiredFeatures = ["xsfvfexpa"] in {
  defm sf_vfexpa : RVVVFEXPBuiltinSet<[["v", "v", "vv"]], "xf">;
}

let RequiredFeatures = ["xsfvfexpa64e"] in {
  defm sf_vfexpa : RVVVFEXPBuiltinSet<[["v", "v", "vv"]], "d">;
}

let UnMaskedPolicyScheme = HasPolicyOperand in
  let RequiredFeatures = ["xsfvfwmaccqqq"] in
    defm sf_vfwmacc_4x4x4 : RVVVFWMACCBuiltinSet<[["", "Fw", "FwFwSvv"]]>;

let UnMaskedPolicyScheme = HasPassthruOperand, RequiredFeatures = ["xsfvfnrclipxfqf"] in {
let ManualCodegen = [{
  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
}] in {
  let HasFRMRoundModeOp = true in {
    defm sf_vfnrclip_x_f_qf : RVVVFNRCLIPBuiltinSet<"v", "vFqfu", "c">;
    defm sf_vfnrclip_xu_f_qf : RVVVFNRCLIPBuiltinSet<"Uv", "UvFqfu", "c">;
  }
  defm sf_vfnrclip_x_f_qf : RVVVFNRCLIPBuiltinSet<"v", "vFqf", "c">;
  defm sf_vfnrclip_xu_f_qf : RVVVFNRCLIPBuiltinSet<"Uv", "UvFqf", "c">;
}
}

multiclass RVVSFTileLoadStoreBuiltinSet<list<string> types,
                                        list<string> RequiredFeatures = []> {
  let OverloadedName = NAME,
      Name = NAME,
      IRName = NAME,
      Log2LMUL = [0],
      HasMasked = false,
      ManualCodegen = [{IntrinsicTypes = {Ops.back()->getType()};}] in
    foreach type = types in {
      let RequiredFeatures = !listconcat(RequiredFeatures,
                                         !cond(!eq(type, "x"): ["zvfhmin"],
                                               !eq(type, "y"): ["zvfbfmin"],
                                               true:           []<string>)) in {
        def : RVVBuiltin<"e", "0zPCe", type>;
        if !not(IsFloat<type>.val) then
          def : RVVBuiltin<"Ue", "0zPCUe", type>;
      }
    }
}

multiclass RVVSFTileMoveBuiltinSet<list<list<string>> suffixes_prototypes,
                                   list<int> intrinsic_types,
                                   string type,
                                   list<string> RequiredFeatures = []> {
  foreach sp = suffixes_prototypes in
    let RequiredFeatures = !listconcat(RequiredFeatures,
                                       !cond(!eq(type, "x"): ["zvfhmin"],
                                             !eq(type, "y"): ["zvfbfmin"],
                                             true:           []<string>)),
        SupportOverloading = false,
        HasMasked = false,
        Name = NAME,
        IRName = NAME,
        HasVL = true,
        Log2LMUL = [3],
        IntrinsicTypes = intrinsic_types in
      def : RVVBuiltin<sp[0], sp[1], type>;
}

multiclass RVVSFTileMoveVTBuiltinSet<list<string> RequiredFeatures = []> {
  foreach type = ["c", "s", "i", "l"] in
    defm NAME :
        RVVSFTileMoveBuiltinSet<[["v", "vz"], ["Uv", "Uvz"]], [-1], type,
                                RequiredFeatures>;
  foreach type = ["x", "y", "f", "d"] in
    defm NAME :
        RVVSFTileMoveBuiltinSet<[["v", "vz"]], [-1], type, RequiredFeatures>;
}

multiclass RVVSFTileMoveTVBuiltinSet<list<string> RequiredFeatures = []> {
  let SupportOverloading = true, OverloadedName = NAME in {
    foreach type = ["c", "s", "i", "l"] in
      defm NAME :
          RVVSFTileMoveBuiltinSet<[["v", "0zv"], ["Uv", "0zUv"]], [1], type,
                                  RequiredFeatures>;
    foreach type = ["x", "y", "f", "d"] in
      defm NAME :
          RVVSFTileMoveBuiltinSet<[["v", "0zv"]], [1], type, RequiredFeatures>;
  }
}

multiclass RVVOp0Op1Op2BuiltinSet<string intrinsic_name, string type_range,
                                  list<list<string>> suffixes_prototypes>
    : RVVBuiltinSet<intrinsic_name, type_range, suffixes_prototypes, [0, 1, 2]>;

multiclass RVVSFMatMulBuiltinSet<string prototype, string suffix,
                                 string type_range, list<int> widens> {
  foreach widen = widens in
    let OverloadedName = NAME,
        TWiden = widen,
        HasVL = false,
        Log2LMUL = [3],
        HasMasked = false in
      defm NAME : RVVOp0Op1Op2BuiltinSet<NAME, type_range,
          [[!strconcat("w", !cast<string>(widen)), suffix, prototype]]>;
}

multiclass RVVSFMatMulFloatBuiltinSet<string name, string prototype, string suffix,
                                      list<string> type_range, int widen> {
  // Currently the XSfmm spec doesn't support w8.
  foreach type = type_range in
    let OverloadedName = name # !strconcat("_w", !cast<string>(widen)),
        TWiden = widen,
        HasVL = false,
        Log2LMUL = [3],
        Name = name # "_" # !strconcat("w", !cast<string>(widen)),
        HasMasked = false in
      defm : RVVOp0Op1BuiltinSet<name, type, [["", suffix, prototype]]>;
}

multiclass RVVSFVTZeroBuiltinSet {
  let SupportOverloading = false,
      HasVL = false,
      HasMasked = false,
      Name = NAME,
      IRName = NAME,
      Log2LMUL = [0] in
    defm : RVVOp0BuiltinSet<NAME, "i", [["", "", "0Kzzzzz"]]>;
}

multiclass RVVSFVTDiscardBuiltinSet {
  let SupportOverloading = false,
      HasVL = false,
      HasMasked = false,
      Name = NAME,
      IRName = NAME,
      Log2LMUL = [0] in
    defm : RVVBuiltinSet<NAME, "i", [["", "", "0"]], []>;
}

let RequiredFeatures = ["xsfmmbase"] in {
  let SupportOverloading = false,
      HasVL = false,
      HasMasked = false,
      Log2LMUL = [0],
      ManualCodegen = [{IntrinsicTypes = {ResultType};}] in // Set XLEN type
  {
    // let HasBuiltinAlias = false in
    def sf_vsettnt : RVVBuiltin<"", "zzKzKz", "i">;
    def sf_vsettm  : RVVBuiltin<"", "zzKzKz", "i">;
    let IRName = "sf_vsettnt" in
      def sf_vsettn  : RVVBuiltin<"", "zzKzKz", "i">;
    def sf_vsettk  : RVVBuiltin<"", "zzKzKz", "i">;
  }
  defm sf_vtzero_t : RVVSFVTZeroBuiltinSet;
  defm sf_vtdiscard : RVVSFVTDiscardBuiltinSet;
}

defm sf_vtmv_v_t : RVVSFTileMoveVTBuiltinSet<["xsfmmbase"]>;
defm sf_vtmv_t_v : RVVSFTileMoveTVBuiltinSet<["xsfmmbase"]>;

defm sf_vlte8  : RVVSFTileLoadStoreBuiltinSet<["c"], ["xsfmmbase"]>;
defm sf_vlte16 : RVVSFTileLoadStoreBuiltinSet<["s", "x", "y"], ["xsfmmbase"]>;
defm sf_vlte32 : RVVSFTileLoadStoreBuiltinSet<["i", "f"], ["xsfmmbase"]>;
defm sf_vlte64 : RVVSFTileLoadStoreBuiltinSet<["l", "d"], ["xsfmmbase"]>;

defm sf_vste8  : RVVSFTileLoadStoreBuiltinSet<["c"], ["xsfmmbase"]>;
defm sf_vste16 : RVVSFTileLoadStoreBuiltinSet<["s", "x", "y"], ["xsfmmbase"]>;
defm sf_vste32 : RVVSFTileLoadStoreBuiltinSet<["i", "f"], ["xsfmmbase"]>;
defm sf_vste64 : RVVSFTileLoadStoreBuiltinSet<["l", "d"], ["xsfmmbase"]>;

let RequiredFeatures = ["xsfmm32a8i"] in {
  defm sf_mm_u_u   : RVVSFMatMulBuiltinSet<"0KzUvUvzzz", "UvUv", "c", [4]>;
  defm sf_mm_s_u   : RVVSFMatMulBuiltinSet<"0KzvUvzzz", "vUv", "c", [4]>;
  defm sf_mm_u_s   : RVVSFMatMulBuiltinSet<"0KzUvvzzz", "Uvv", "c", [4]>;
  defm sf_mm_s_s   : RVVSFMatMulBuiltinSet<"0Kzvvzzz", "vv", "c", [4]>;

}

let RequiredFeatures = ["xsfmm32a16f"] in
  defm : RVVSFMatMulFloatBuiltinSet<"sf_mm_f_f", "0Kzvvzzz", "v", ["x", "y"], 2>;

let RequiredFeatures = ["xsfmm32a32f"] in
  defm : RVVSFMatMulFloatBuiltinSet<"sf_mm_f_f", "0Kzvvzzz", "v", ["f"], 1>;

let RequiredFeatures = ["xsfmm32a8f"] in
  foreach e1 = [5, 4] in
    foreach e2 = [5, 4] in
      let OverloadedName = "sf_mm_e" # e1 # "m" # !sub(7, e1) # "_e" # e2 # "m" # !sub(7, e2) in
        defm : RVVSFMatMulFloatBuiltinSet<
            "sf_mm_e" # e1 # "m" # !sub(7, e1) # "_e" # e2 # "m" # !sub(7, e2),
            "0KzUvUvzzz", "UvUv", ["c"], 4>;

let RequiredFeatures = ["xsfmm64a64f"] in
  defm : RVVSFMatMulFloatBuiltinSet<"sf_mm_f_f", "0Kzvvzzz", "v", ["d"], 1>;
