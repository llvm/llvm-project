//===--- SymbolName.h - Clang refactoring library ----------*- C++ -*------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_SYMBOL_NAME_H
#define LLVM_CLANG_TOOLING_REFACTOR_SYMBOL_NAME_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace clang {

class LangOptions;

namespace tooling {

/// \brief A name of a declaration that's used in the refactoring process.
///
/// Names can be composed of multiple string, to account for things like
/// Objective-C selectors.
class OldSymbolName {
public:
  OldSymbolName() {}

  /// \brief Creates a \c SymbolName by decomposing the given \p Name using
  /// language specific logic.
  OldSymbolName(StringRef Name, const LangOptions &LangOpts);
  OldSymbolName(StringRef Name, bool IsObjectiveCSelector);
  explicit OldSymbolName(ArrayRef<StringRef> Name);

  OldSymbolName(OldSymbolName &&) = default;
  OldSymbolName &operator=(OldSymbolName &&) = default;

  OldSymbolName(const OldSymbolName &) = default;
  OldSymbolName &operator=(const OldSymbolName &) = default;

  bool empty() const { return Strings.empty(); }

  /// \brief Returns the number of the strings that make up the given name.
  size_t size() const { return Strings.size(); }

  /// \brief Returns the string at the given index.
  StringRef operator[](size_t I) const { return Strings[I]; }

  ArrayRef<std::string> strings() const { return Strings; }

  bool containsEmptyPiece() const {
    for (const auto &String : Strings) {
      if (String.empty())
        return true;
    }
    return false;
  }

  void print(raw_ostream &OS) const;

private:
  std::vector<std::string> Strings;
};

raw_ostream &operator<<(raw_ostream &OS, const OldSymbolName &N);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_SYMBOL_NAME_H
