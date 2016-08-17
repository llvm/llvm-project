//===--- APINotesOptions.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

namespace clang {

/// APINotesOptions - Track various options which control how API
/// notes are found and handled.
class APINotesOptions {
public:
  /// The set of search paths where we API notes can be found for
  /// particular modules.
  ///
  /// The API notes in this directory are stored as
  /// <ModuleName>.apinotes or <ModuleName>.apinotesc, and are only
  /// applied when building the module <ModuleName>.
  std::vector<std::string> ModuleSearchPaths;
};

}  // end namespace clang

#endif
