//===--- APINotesOptions.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the APINotesOptions class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_APINOTES_APINOTESOPTIONS_H
#define LLVM_CLANG_APINOTES_APINOTESOPTIONS_H

#include <string>
#include <vector>
#include "llvm/Support/VersionTuple.h"

namespace clang {

/// APINotesOptions - Track various options which control how API
/// notes are found and handled.
class APINotesOptions {
public:
  /// The Swift version which should be used for API notes.
  llvm::VersionTuple SwiftVersion;

  /// The set of search paths where we API notes can be found for
  /// particular modules.
  ///
  /// The API notes in this directory are stored as <ModuleName>.apinotes,
  /// and are only applied when building the module <ModuleName>.
  std::vector<std::string> ModuleSearchPaths;
};

}  // end namespace clang

#endif
