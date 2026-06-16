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

class RISCVXCVBuiltin<string prototype, string features = ""> : TargetBuiltin {
  let Spellings = ["__builtin_riscv_cv_" # NAME];
  let Prototype = prototype;
  let Features = features;
}

let Attributes = [NoThrow, Const] in {
//===----------------------------------------------------------------------===//
// XCVbitmanip extension.
//===----------------------------------------------------------------------===//
def bitmanip_extract  : RISCVXCVBuiltin<"int(unsigned int, unsigned short)", "xcvbitmanip">;
def bitmanip_extractu : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned short)", "xcvbitmanip">;
def bitmanip_insert   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned short, unsigned int)", "xcvbitmanip">;
def bitmanip_bclr     : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned short)", "xcvbitmanip">;
def bitmanip_bset     : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned short)", "xcvbitmanip">;
def bitmanip_ff1      : RISCVXCVBuiltin<"unsigned char(unsigned int)", "xcvbitmanip">;
def bitmanip_fl1      : RISCVXCVBuiltin<"unsigned char(unsigned int)", "xcvbitmanip">;
def bitmanip_clb      : RISCVXCVBuiltin<"unsigned char(unsigned int)", "xcvbitmanip">;
def bitmanip_cnt      : RISCVXCVBuiltin<"unsigned char(unsigned int)", "xcvbitmanip">;
def bitmanip_ror      : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvbitmanip">;
def bitmanip_bitrev   : RISCVXCVBuiltin<"unsigned int(unsigned int, _Constant unsigned char, _Constant unsigned char)", "xcvbitmanip">;

//===----------------------------------------------------------------------===//
// XCValu extension.
//===----------------------------------------------------------------------===//
def alu_sle   : RISCVXCVBuiltin<"int(int, int)", "xcvalu">;
def alu_sleu  : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvalu">;
def alu_exths : RISCVXCVBuiltin<"int(int)", "xcvalu">;
def alu_exthz : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvalu">;
def alu_extbs : RISCVXCVBuiltin<"int(int)", "xcvalu">;
def alu_extbz : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvalu">;

def alu_clip   : RISCVXCVBuiltin<"int(int, int)", "xcvalu">;
def alu_clipu  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvalu">;
def alu_addN   : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_adduN  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_addRN  : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_adduRN : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_subN   : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_subuN  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_subRN  : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_subuRN : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
} // Attributes = [NoThrow, Const]
