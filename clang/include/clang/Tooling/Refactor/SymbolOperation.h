//===--- SymbolOperation.h - -------------------------------*- C++ -*------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_SYMBOL_OPERATION_H
#define LLVM_CLANG_TOOLING_REFACTOR_SYMBOL_OPERATION_H

#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Refactor/RenamedSymbol.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace clang {

class ASTContext;
class NamedDecl;

namespace tooling {

/// \brief A refactoring operation that deals with occurrences of symbols.
class SymbolOperation {
  /// Contains the symbols that are required for this operation.
  SmallVector<rename::Symbol, 4> Symbols;

  /// Maps from a USR to an index in the \c Symbol array.
  /// Contains all of the USRs that correspond to the declarations which use
  /// the symbols in this operation.
  llvm::StringMap<unsigned> USRToSymbol;

  /// True if all the symbols in this operation occur only in the translation
  /// unit that defines them.
  bool IsLocal;

  /// The declaration whose implementation is needed for the correct initiation
  /// of a symbol operation.
  const NamedDecl *DeclThatRequiresImplementationTU;

public:
  SymbolOperation(const NamedDecl *FoundDecl, ASTContext &Context);

  SymbolOperation(SymbolOperation &&) = default;
  SymbolOperation &operator=(SymbolOperation &&) = default;

  /// Return the symbol that corresponds to the given USR, or null if this USR
  /// isn't interesting from the perspective of this operation.
  const rename::Symbol *getSymbolForUSR(StringRef USR) const {
    auto It = USRToSymbol.find(USR);
    if (It != USRToSymbol.end())
      return &Symbols[It->getValue()];
    return nullptr;
  }

  /// The symbols that this operation is working on.
  ///
  /// Symbol operations, like rename, usually just work on just one symbol.
  /// However, there are certain language constructs that require more than
  /// one symbol in order for them to be renamed correctly. Property
  /// declarations in Objective-C are the perfect example: in addition to the
  /// actual property, renaming has to rename the corresponding getters and
  /// setters, as well as the backing ivar.
  ArrayRef<rename::Symbol> symbols() const { return Symbols; }

  /// True if all the symbols in this operation occur only in the translation
  /// unit that defines them.
  bool isLocal() const { return IsLocal; }

  /// True if the declaration that was found in the initial TU needs to be
  /// examined in the TU that implemented it.
  bool requiresImplementationTU() const {
    return DeclThatRequiresImplementationTU;
  }

  /// Returns the declaration whose implementation is needed for the correct
  /// initiation of a symbol operation.
  const NamedDecl *declThatRequiresImplementationTU() const {
    return DeclThatRequiresImplementationTU;
  }
};

/// Return true if the given declaration corresponds to a local symbol.
bool isLocalSymbol(const NamedDecl *D, const LangOptions &LangOpts);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_SYMBOL_OPERATION_H
