//==- BuiltinsRISCVXAndes.td - RISC-V Andes Builtin database -----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Andes-specific builtin function database.  Users of
// this file must define the BUILTIN macro to make use of this information.
//
//===----------------------------------------------------------------------===//

class RISCVXAndesBuiltin<string prototype, string features = ""> : TargetBuiltin {
  let Spellings = ["__builtin_riscv_nds_" # NAME];
  let Prototype = prototype;
  let Features = features;
}

let Attributes = [NoThrow, Const] in {
//===----------------------------------------------------------------------===//
// XAndesBFHCvt extension.
//===----------------------------------------------------------------------===//

def fcvt_s_bf16 : RISCVXAndesBuiltin<"float(__bf16)", "xandesbfhcvt">;
def fcvt_bf16_s : RISCVXAndesBuiltin<"__bf16(float)", "xandesbfhcvt">;
} // Attributes = [NoThrow, Const]
