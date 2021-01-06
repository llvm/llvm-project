//===- PreprocessorOptions.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the PreprocessorOptions class, which
/// is the class for all preprocessor options.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_PREPROCESSOROPTIONS_H
#define LLVM_FLANG_PREPROCESSOROPTIONS_H

#include "llvm/ADT/StringRef.h"

namespace Fortran::frontend {

/// This class is used for passing the various options used
/// in preprocessor initialization to the parser options.
class PreprocessorOptions {
public:
  std::vector<std::pair<std::string, /*isUndef*/ bool>> macros;

public:
  PreprocessorOptions() {}

  void addMacroDef(llvm::StringRef name) {
    macros.emplace_back(std::string(name), false);
  }

  void addMacroUndef(llvm::StringRef name) {
    macros.emplace_back(std::string(name), true);
  }
};

} // namespace Fortran::frontend

#endif // LLVM_FLANG_PREPROCESSOROPTIONS_H