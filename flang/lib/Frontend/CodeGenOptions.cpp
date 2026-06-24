//===--- CodeGenOptions.cpp -----------------------------------------------===//
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

#include "flang/Frontend/CodeGenOptions.h"
#include <optional>
#include <string.h>

namespace Fortran::frontend {

CodeGenOptions::CodeGenOptions() {
#define CODEGENOPT(Name, Bits, Default) Name = Default;
#define ENUM_CODEGENOPT(Name, Type, Bits, Default) set##Name(Default);
#include "flang/Frontend/CodeGenOptions.def"
}

std::optional<llvm::CodeModel::Model> getCodeModel(llvm::StringRef string) {
  return llvm::StringSwitch<std::optional<llvm::CodeModel::Model>>(string)
      .Case("tiny", llvm::CodeModel::Model::Tiny)
      .Case("small", llvm::CodeModel::Model::Small)
      .Case("kernel", llvm::CodeModel::Model::Kernel)
      .Case("medium", llvm::CodeModel::Model::Medium)
      .Case("large", llvm::CodeModel::Model::Large)
      .Default(std::nullopt);
}

} // end namespace Fortran::frontend
