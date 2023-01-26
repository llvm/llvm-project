//===--- LoweringOptions.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/LoweringOptions.h"

namespace Fortran::lower {

LoweringOptions::LoweringOptions() : MathOptions{} {
#define LOWERINGOPT(Name, Bits, Default) Name = Default;
#define ENUM_LOWERINGOPT(Name, Type, Bits, Default) set##Name(Default);
#include "flang/Lower/LoweringOptions.def"
}

} // namespace Fortran::lower
