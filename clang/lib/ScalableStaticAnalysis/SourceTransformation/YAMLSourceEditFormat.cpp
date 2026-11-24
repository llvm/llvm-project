//===- YAMLSourceEditFormat.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/SourceTransformation/YAMLSourceEditFormat.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ssaf;

llvm::Error ssaf::writeYAMLSourceEdits(
    const clang::tooling::TranslationUnitReplacements &Doc,
    llvm::StringRef Path) {
  std::error_code EC;
  llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::OF_None);
  if (EC)
    return llvm::createStringError(EC, "failed to open '" + Path + "'");

  // llvm::yaml::Output's stream operator binds to a non-const reference.
  clang::tooling::TranslationUnitReplacements Mutable = Doc;
  llvm::yaml::Output YAMLOut(OS);
  YAMLOut << Mutable;

  return llvm::Error::success();
}
