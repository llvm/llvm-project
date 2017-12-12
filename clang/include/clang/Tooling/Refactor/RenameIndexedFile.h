//===--- RenameIndexedFile.h - -----------------------------*- C++ -*------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_RENAME_INDEXED_FILE_H
#define LLVM_CLANG_TOOLING_REFACTOR_RENAME_INDEXED_FILE_H

#include "clang/Basic/LLVM.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Refactor/RenamedSymbol.h"
#include "clang/Tooling/Refactor/SymbolName.h"
#include "llvm/ADT/ArrayRef.h"

namespace clang {
namespace tooling {

class RefactoringOptionSet;

namespace rename {

/// An already known occurrence of the symbol that's being renamed.
struct IndexedOccurrence {
  /// The location of this occurrence in the indexed file.
  unsigned Line, Column;
  enum OccurrenceKind {
    IndexedSymbol,
    IndexedObjCMessageSend,
    InclusionDirective
  };
  OccurrenceKind Kind;
};

struct IndexedSymbol {
  SymbolName Name;
  std::vector<IndexedOccurrence> IndexedOccurrences;
  /// Whether this symbol is an Objective-C selector.
  bool IsObjCSelector;
  /// If true, indexed file renamer will look for matching textual occurrences
  /// in string literal tokens.
  bool SearchForStringLiteralOccurrences;

  IndexedSymbol(SymbolName Name,
                std::vector<IndexedOccurrence> IndexedOccurrences,
                bool IsObjCSelector,
                bool SearchForStringLiteralOccurrences = false)
      : Name(std::move(Name)),
        IndexedOccurrences(std::move(IndexedOccurrences)),
        IsObjCSelector(IsObjCSelector),
        SearchForStringLiteralOccurrences(SearchForStringLiteralOccurrences) {}
  IndexedSymbol(IndexedSymbol &&Other) = default;
  IndexedSymbol &operator=(IndexedSymbol &&Other) = default;
};

/// Consumes the \c SymbolOccurrences found by \c IndexedFileOccurrenceProducer.
class IndexedFileOccurrenceConsumer {
public:
  virtual ~IndexedFileOccurrenceConsumer() {}
  virtual void handleOccurrence(const SymbolOccurrence &Occurrence,
                                SourceManager &SM,
                                const LangOptions &LangOpts) = 0;
};

/// Finds the renamed \c SymbolOccurrences in an already indexed files.
class IndexedFileOccurrenceProducer final : public PreprocessorFrontendAction {
  bool IsMultiPiece;
  ArrayRef<IndexedSymbol> Symbols;
  IndexedFileOccurrenceConsumer &Consumer;
  const RefactoringOptionSet *Options;

public:
  IndexedFileOccurrenceProducer(ArrayRef<IndexedSymbol> Symbols,
                                IndexedFileOccurrenceConsumer &Consumer,
                                const RefactoringOptionSet *Options = nullptr);

private:
  void ExecuteAction() override;
};

} // end namespace rename
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_RENAME_INDEXED_FILE_H
