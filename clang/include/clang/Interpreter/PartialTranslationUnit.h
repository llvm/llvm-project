//===--- Transaction.h - Incremental Compilation and Execution---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities tracking the incrementally processed pieces of
// code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_PARTIALTRANSLATIONUNIT_H
#define LLVM_CLANG_INTERPRETER_PARTIALTRANSLATIONUNIT_H

#include "clang/Basic/FileEntry.h"
#include <memory>

namespace llvm {
class Module;
}

namespace clang {

class IdentifierInfo;
class MacroDirective;
class TranslationUnitDecl;

/// The class keeps track of various objects created as part of processing
/// incremental inputs.
struct PartialTranslationUnit {
  TranslationUnitDecl *TUPart = nullptr;

  /// The llvm IR produced for the input.
  std::unique_ptr<llvm::Module> TheModule;
  bool operator==(const PartialTranslationUnit &other) {
    return other.TUPart == TUPart && other.TheModule == TheModule;
  }

  ///\brief Each macro pair (is this the same as for decls?)came
  /// through different interface at
  /// different time. We are being conservative and we want to keep all the
  /// call sequence that originally occurred in clang.
  ///
  struct MacroDirectiveInfo {
    // We need to store both the IdentifierInfo and the MacroDirective
    // because the Preprocessor stores the macros in a DenseMap<II, MD>.
    IdentifierInfo *II;
    const MacroDirective *MD;
    MacroDirectiveInfo(IdentifierInfo *II, const MacroDirective *MD)
        : II(II), MD(MD) {}
    inline bool operator==(const MacroDirectiveInfo &rhs) const {
      return II == rhs.II && MD == rhs.MD;
    }
    inline bool operator!=(const MacroDirectiveInfo &rhs) const {
      return !operator==(rhs);
    }
  };
  // Intentionally use struct instead of pair because we don't need default
  // init.
  // Add macro decls to be able to revert them for error recovery.
  typedef llvm::SmallVector<MacroDirectiveInfo, 2> MacroDirectiveInfoQueue;

  ///\brief All seen macros.
  ///
  MacroDirectiveInfoQueue MacroDirectiveInfos;

  /// Files that were #included during this PTU's parsing.
  /// Used to reset HeaderFileInfo on Undo.
  std::vector<FileEntryRef> IncludedFiles;
};
} // namespace clang

#endif // LLVM_CLANG_INTERPRETER_PARTIALTRANSLATIONUNIT_H
