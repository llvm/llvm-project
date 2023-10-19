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
                             list<int> intrinsic_types, bit UseGPR> {
  foreach r = range in
    let RequiredFeatures = !if(!and(UseGPR, !eq(r, "l")),
                               ["Xsfvcp", "RV64"], ["Xsfvcp"]) in
      defm : VCIXBuiltinSet<NAME, NAME, "Uv", prototype, r, intrinsic_types>;
}

multiclass RVVVCIXBuiltinSetWVType<list<string> range, string prototype,
                             list<int> intrinsic_types, bit UseGPR> {
  foreach r = range in
    let RequiredFeatures = !if(!and(UseGPR, !eq(r, "l")),
                               ["Xsfvcp", "RV64"], ["Xsfvcp"]) in
      // These intrinsics don't have any vector types in the output and inputs,
      // but we still need to add vetvli for them. So we encode different
      // VTYPE into the intrinsic names, and then will know which vsetvli is
      // correct.
      foreach s = VCIXSuffix<r>.suffix in
        // Since we already encode the Vtype into the name, so just set
        // Log2LMUL to zero.  Otherwise the RISCVVEmitter will expand
        // lots of redundant intrinsic but have same names.
        let Log2LMUL = [0] in
          def : VCIXBuiltinSet<NAME # "_u" # s, NAME # "_e" # s,
                               "", prototype, r, intrinsic_types>;
}

let SupportOverloading = false in {
  defm sf_vc_x_se  : RVVVCIXBuiltinSetWVType<["c", "s", "i", "l"], "0KzKzKzUe", [0, 3], UseGPR=1>;
  defm sf_vc_i_se  : RVVVCIXBuiltinSetWVType<["c", "s", "i", "l"], "0KzKzKzKz", [2, 3], UseGPR=0>;
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
