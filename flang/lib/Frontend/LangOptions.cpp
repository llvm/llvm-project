//===------ LangOptions.cpp -----------------------------------------------===//
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

#include "flang/Frontend/LangOptions.h"
#include <string.h>

namespace Fortran::frontend {

LangOptions::LangOptions() {
#define LANGOPT(Name, Bits, Default) Name = Default;
#define ENUM_LANGOPT(Name, Type, Bits, Default) set##Name(Default);
#include "flang/Frontend/LangOptions.def"
}

} // end namespace Fortran::frontend
