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
// XAndesPerf extension.
//===----------------------------------------------------------------------===//

// https://github.com/andestech/andes-intrinsic-doc/blob/main/andes-scalar-intrinsic.md

let Features = "xandesperf,32bit" in {
  def ffb_32 : RISCVXAndesBuiltin<"int32_t(uint32_t, uint32_t)">;
  def ffzmism_32 : RISCVXAndesBuiltin<"int32_t(uint32_t, uint32_t)">;
  def ffmism_32 : RISCVXAndesBuiltin<"int32_t(uint32_t, uint32_t)">;
  def flmism_32 : RISCVXAndesBuiltin<"int32_t(uint32_t, uint32_t)">;
}

let Features = "xandesperf,64bit" in {
  def ffb_64 : RISCVXAndesBuiltin<"int64_t(uint64_t, uint64_t)">;
  def ffzmism_64 : RISCVXAndesBuiltin<"int64_t(uint64_t, uint64_t)">;
  def ffmism_64 : RISCVXAndesBuiltin<"int64_t(uint64_t, uint64_t)">;
  def flmism_64 : RISCVXAndesBuiltin<"int64_t(uint64_t, uint64_t)">;
}

//===----------------------------------------------------------------------===//
// XAndesBFHCvt extension.
//===----------------------------------------------------------------------===//

def fcvt_s_bf16 : RISCVXAndesBuiltin<"float(__bf16)", "xandesbfhcvt">;
def fcvt_bf16_s : RISCVXAndesBuiltin<"__bf16(float)", "xandesbfhcvt">;
} // Attributes = [NoThrow, Const]
