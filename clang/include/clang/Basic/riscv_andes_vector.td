//==--- riscv_andes_vector.td - RISC-V Andes Builtin function list --------===//
//
//  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the builtins for RISC-V Andes Vector Extension. See:
//
//     https://github.com/andestech/andes-vector-intrinsic-doc
//
//===----------------------------------------------------------------------===//

include "riscv_vector_common.td"

//===----------------------------------------------------------------------===//
// Instruction definitions
//===----------------------------------------------------------------------===//

// Andes Vector BFLOAT16 Conversion Extension (XAndesVBFHCvt)

let RequiredFeatures = ["xandesvbfhcvt"],
    Log2LMUL = [-2, -1, 0, 1, 2],
    HasMasked = false,
    UnMaskedPolicyScheme = HasPassthruOperand in {
  def nds_vfwcvt_s_bf16 : RVVConvBuiltin<"Fw", "Fwv", "y", "nds_vfwcvt_s">;

  let ManualCodegen = [{
    {
      // LLVM intrinsic
      // Unmasked: (passthru, op0, frm, vl)
      SmallVector<llvm::Value*, 4> Operands;
      bool HasMaskedOff = !(PolicyAttrs & RVV_VTA);
      bool HasRoundModeOp = HasMaskedOff ? Ops.size() == 4 : Ops.size() == 3;

      unsigned Offset = HasMaskedOff ? 1 : 0;

      if (!HasMaskedOff)
        Operands.push_back(llvm::PoisonValue::get(ResultType));
      else
        Operands.push_back(Ops[0]);

      Operands.push_back(Ops[Offset]); // op0

      if (HasRoundModeOp) {
        Operands.push_back(Ops[Offset + 1]); // frm
        Operands.push_back(Ops[Offset + 2]); // vl
      } else {
        Operands.push_back(ConstantInt::get(Ops[Offset + 1]->getType(), 7)); // frm
        Operands.push_back(Ops[Offset + 1]); // vl
      }

      IntrinsicTypes = {ResultType, Ops[Offset]->getType(),
                        Operands.back()->getType()};
      llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
      return Builder.CreateCall(F, Operands, "");
    }
  }] in {
    let HasFRMRoundModeOp = 1, Name = "nds_vfncvt_bf16_s", IRName = "nds_vfncvt_bf16_s" in
      def nds_vfncvt_bf16_s_rm : RVVConvBuiltin<"v", "vFwu", "y", "nds_vfncvt_bf16">;

    def nds_vfncvt_bf16_s : RVVConvBuiltin<"v", "vFw", "y", "nds_vfncvt_bf16">;
  }
}

// Andes Vector INT4 Load Extension (XAndesVSIntLoad)

let SupportOverloading = false,
    UnMaskedPolicyScheme = HasPassthruOperand in {
  multiclass RVVVLN8Builtin {
    let Name = NAME # "_v",
        IRName = "nds_vln",
        MaskedIRName = "nds_vln_mask",
        OverloadedName = NAME in
      def :  RVVOutOp0Builtin<"v", "vPC0", "c">;
  }
}

let SupportOverloading = false,
    UnMaskedPolicyScheme = HasPassthruOperand in {
  multiclass RVVVLNU8Builtin {
    let Name = NAME # "_v",
        IRName = "nds_vlnu",
        MaskedIRName = "nds_vlnu_mask",
        OverloadedName = NAME in
      def : RVVOutOp0Builtin<"Uv", "UvPC0", "c">;
  }
}

let RequiredFeatures = ["xandesvsintload"] in {
defm nds_vln8  : RVVVLN8Builtin;
defm nds_vlnu8 : RVVVLNU8Builtin;
}

// Andes Vector Packed FP16 Extension (XAndesVPackFPH)

multiclass RVVFPMAD {
  let Log2LMUL = [-2, -1, 0, 1, 2, 3],
      OverloadedName = NAME in {
    defm NAME : RVVOutOp1BuiltinSet<NAME, "x", [["vf", "v", "vvf"]]>;

    let HasFRMRoundModeOp = true in
      defm NAME : RVVOutOp1BuiltinSet<NAME, "x", [["vf", "v", "vvfu"]]>;
  }
}

let RequiredFeatures = ["xandesvpackfph"],
    UnMaskedPolicyScheme = HasPassthruOperand in {
let ManualCodegen = [{
  {
    // LLVM intrinsic
    // Unmasked: (passthru, op0, op1, round_mode, vl)
    // Masked:   (passthru, vector_in, vector_in/scalar_in, mask, frm, vl, policy)

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

    IntrinsicTypes = {ResultType, Ops[Offset + 1]->getType(),
                      Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
}] in {
    defm nds_vfpmadt : RVVFPMAD;
    defm nds_vfpmadb : RVVFPMAD;
  }
}

// Andes Vector Dot Product Extension (XAndesVDot)

multiclass RVVD4DOT<list<list<string>> i_suffixes_prototypes,
                    list<list<string>> l_suffixes_prototypes> {
  let RequiredFeatures = ["xandesvdot"],
      UnMaskedPolicyScheme = HasPolicyOperand,
      HasMaskedOffOperand = false,
      Log2LMUL = [-1, 0, 1, 2, 3],
      OverloadedName = NAME in {
    defm NAME : RVVOutOp1Op2BuiltinSet<NAME, "i", i_suffixes_prototypes>;
    defm NAME : RVVOutOp1Op2BuiltinSet<NAME, "l", l_suffixes_prototypes>;
  }
}

defm nds_vd4dots  : RVVD4DOT<[["vv", "v", "vv(FixedSEW:8)v(FixedSEW:8)v"]],
                             [["vv", "v", "vv(FixedSEW:16)v(FixedSEW:16)v"]]>;
defm nds_vd4dotu  : RVVD4DOT<[["vv", "Uv", "UvUv(FixedSEW:8)Uv(FixedSEW:8)Uv"]],
                             [["vv", "Uv", "UvUv(FixedSEW:16)Uv(FixedSEW:16)Uv"]]>;
defm nds_vd4dotsu : RVVD4DOT<[["vv", "v", "vv(FixedSEW:8)v(FixedSEW:8)Uv"]],
                             [["vv", "v", "vv(FixedSEW:16)v(FixedSEW:16)Uv"]]>;
