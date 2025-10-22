//==- BuiltinsRISCVXMIPS.td - RISC-V MIPS  Builtin database    ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MIPS-specific builtin function database.  Users of
// this file must define the BUILTIN macro to make use of this information.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MIPS execution control extensions.
//===----------------------------------------------------------------------===//
let Attributes = [NoThrow, Const] in {
def mips_pause : RISCVBuiltin<"void()", "xmipsexectl">;
def mips_ehb   : RISCVBuiltin<"void()", "xmipsexectl">;
def mips_ihb   : RISCVBuiltin<"void()", "xmipsexectl">;
}
