//==- BuiltinsRISCVXCV.td - RISC-V CORE-V Builtin database    ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CORE-V-specific builtin function database.  Users of
// this file must define the BUILTIN macro to make use of this information.
//
//===----------------------------------------------------------------------===//

class RISCXCVBuiltin<string prototype, string features = ""> : TargetBuiltin {
  let Spellings = ["__builtin_riscv_cv_" # NAME];
  let Prototype = prototype;
  let Features = features;
}

let Attributes = [NoThrow, Const] in {
//===----------------------------------------------------------------------===//
// XCValu extension.
//===----------------------------------------------------------------------===//
def alu_slet  : RISCXCVBuiltin<"int(int, int)", "xcvalu">;
def alu_sletu : RISCXCVBuiltin<"int(unsigned int, unsigned int)", "xcvalu">;
def alu_exths : RISCXCVBuiltin<"int(int)", "xcvalu">;
def alu_exthz : RISCXCVBuiltin<"unsigned int(unsigned int)", "xcvalu">;
def alu_extbs : RISCXCVBuiltin<"int(int)", "xcvalu">;
def alu_extbz : RISCXCVBuiltin<"unsigned int(unsigned int)", "xcvalu">;

def alu_clip   : RISCXCVBuiltin<"int(int, int)", "xcvalu">;
def alu_clipu  : RISCXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvalu">;
def alu_addN   : RISCXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_adduN  : RISCXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_addRN  : RISCXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_adduRN : RISCXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_subN   : RISCXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_subuN  : RISCXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_subRN  : RISCXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_subuRN : RISCXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
} // Attributes = [NoThrow, Const]
