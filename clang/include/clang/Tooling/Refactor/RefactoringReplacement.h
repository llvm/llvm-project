//===--- RefactoringReplacement.h - ------------------------*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_REPLACEMENT_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_REPLACEMENT_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Refactoring/Rename/SymbolName.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include <string>

namespace clang {
namespace tooling {

/// \brief Represent a symbol that can be used for an additional refactoring
/// action that associated.
class RefactoringResultAssociatedSymbol {
  SymbolName Name;

public:
  RefactoringResultAssociatedSymbol(SymbolName Name) : Name(std::move(Name)) {}

  const SymbolName &getName() const { return Name; }
};

/// \brief A replacement range.
class RefactoringReplacement {
public:
  SourceRange Range;
  std::string ReplacementString;

  /// \brief Represents a symbol that is contained in the replacement string
  /// of this replacement.
  struct AssociatedSymbolLocation {
    /// These offsets point into the ReplacementString.
    llvm::SmallVector<unsigned, 4> Offsets;
    bool IsDeclaration;

    AssociatedSymbolLocation(ArrayRef<unsigned> Offsets,
                             bool IsDeclaration = false)
        : Offsets(Offsets.begin(), Offsets.end()),
          IsDeclaration(IsDeclaration) {}
  };
  llvm::SmallDenseMap<const RefactoringResultAssociatedSymbol *,
                      AssociatedSymbolLocation>
      SymbolLocations;

  RefactoringReplacement(SourceRange Range) : Range(Range) {}

  RefactoringReplacement(SourceRange Range, StringRef ReplacementString)
      : Range(Range), ReplacementString(ReplacementString.str()) {}
  RefactoringReplacement(SourceRange Range, std::string ReplacementString)
      : Range(Range), ReplacementString(std::move(ReplacementString)) {}

  RefactoringReplacement(SourceRange Range, StringRef ReplacementString,
                         const RefactoringResultAssociatedSymbol *Symbol,
                         const AssociatedSymbolLocation &Loc)
      : Range(Range), ReplacementString(ReplacementString.str()) {
    SymbolLocations.insert(std::make_pair(Symbol, Loc));
  }

  RefactoringReplacement(const FixItHint &Hint) {
    Range = Hint.RemoveRange.getAsRange();
    ReplacementString = Hint.CodeToInsert;
  }

  RefactoringReplacement(RefactoringReplacement &&) = default;
  RefactoringReplacement &operator=(RefactoringReplacement &&) = default;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_REPLACEMENT_H
