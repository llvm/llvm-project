//===--- BuiltinsDirectX.td - DirectX Builtin function database -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

def DxDot2Add : Builtin { 
  let Spellings = ["__builtin_dx_dot2add"];
  let Attributes = [NoThrow, Const];
  let Prototype = "float(_ExtVector<2, _Float16>, _ExtVector<2, _Float16>, float)";
}
