//===-- XrefsDB.h - Interface for symbol-header matching --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_XREFSDB_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_XREFSDB_H

#include "llvm/ADT/StringRef.h"
#include <vector>

namespace clang {
namespace include_fixer {

/// This class provides an interface for finding the header files corresponding
/// to an indentifier in the source code.
class XrefsDB {
public:
  virtual ~XrefsDB() = default;

  /// Search for header files to be included for an identifier.
  /// \param Identifier The identifier being searched for. May or may not be
  ///                   fully qualified.
  /// \returns A list of inclusion candidates, in a format ready for being
  ///          pasted after an #include token.
  // FIXME: Expose the type name so we can also insert using declarations (or
  // fix the usage)
  virtual std::vector<std::string> search(llvm::StringRef Identifier) = 0;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_XREFSDB_H
