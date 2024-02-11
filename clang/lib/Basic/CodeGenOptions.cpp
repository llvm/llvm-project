//===--- CodeGenOptions.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/CodeGenOptions.h"
#include <string.h>

namespace clang {

CodeGenOptions::CodeGenOptions() { resetAllOptions(); }

// FIXME: really reset all codegen options.
void CodeGenOptions::resetAllOptions() { resetNonModularOptions(); }

void CodeGenOptions::resetNonModularOptions() {
#define CODEGENOPT(Name, Bits, Default) Name = Default;
#define ENUM_CODEGENOPT(Name, Type, Bits, Default) set##Name(Default);
// Do not reset AST affecting code generation options.
#define AFFECTING_VALUE_CODEGENOPT(Name, Bits, Default)
#include "clang/Basic/CodeGenOptions.def"

  RelocationModel = llvm::Reloc::PIC_;
  memcpy(CoverageVersion, "408*", 4);
}

}  // end namespace clang
