//===-- XrefsDBManager.h - Managing multiple XrefsDBs -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_XREFSDBMANAGER_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_XREFSDBMANAGER_H

#include "XrefsDB.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace include_fixer {

/// This class provides an interface for finding the header files corresponding
/// to an indentifier in the source code from multiple symbol databases.
class XrefsDBManager {
public:
  void addXrefsDB(std::unique_ptr<XrefsDB> DB) {
    XrefsDBs.push_back(std::move(DB));
  }

  /// Search for header files to be included for an identifier.
  /// \param Identifier The identifier being searched for. May or may not be
  ///                   fully qualified.
  /// \returns A list of inclusion candidates, in a format ready for being
  ///          pasted after an #include token.
  // FIXME: Expose the type name so we can also insert using declarations (or
  // fix the usage)
  // FIXME: Move mapping from SymbolInfo to headers out of
  // XrefsDBManager::search and return SymbolInfos instead of header paths.
  std::vector<std::string> search(llvm::StringRef Identifier) const;

private:
  std::vector<std::unique_ptr<XrefsDB>> XrefsDBs;
};

} // namespace include_fixer
} // namespace clang

#endif
