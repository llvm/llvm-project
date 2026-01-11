//===--- SARIFDiagnostic.h - SARIF Diagnostic Formatting -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a utility class that provides support for constructing a SARIF object
// containing diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_SARIFDIAGNOSTIC_H
#define LLVM_CLANG_FRONTEND_SARIFDIAGNOSTIC_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Sarif.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/DiagnosticRenderer.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

class SARIFDiagnostic : public DiagnosticRenderer {
public:
  SARIFDiagnostic(raw_ostream &OS, const LangOptions &LangOpts,
                  DiagnosticOptions &DiagOpts, SarifDocumentWriter *Writer);

  SARIFDiagnostic &operator=(const SARIFDiagnostic &&) = delete;
  SARIFDiagnostic(SARIFDiagnostic &&) = delete;
  SARIFDiagnostic &operator=(const SARIFDiagnostic &) = delete;
  SARIFDiagnostic(const SARIFDiagnostic &) = delete;

  void writeResult();
  void setLangOptions(const LangOptions& LangOpts);
  void emitInvocation(CompilerInstance& Compiler, bool Successful, StringRef Message);

protected:
  void emitDiagnosticMessage(FullSourceLoc Loc, PresumedLoc PLoc,
                             DiagnosticsEngine::Level Level, StringRef Message,
                             ArrayRef<CharSourceRange> Ranges,
                             DiagOrStoredDiag Diag) override;

  void emitDiagnosticLoc(FullSourceLoc Loc, PresumedLoc PLoc,
                         DiagnosticsEngine::Level Level,
                         ArrayRef<CharSourceRange> Ranges) override;

  void emitCodeContext(FullSourceLoc Loc, DiagnosticsEngine::Level Level,
                       SmallVectorImpl<CharSourceRange> &Ranges,
                       ArrayRef<FixItHint> Hints) override {}

  void emitIncludeLocation(FullSourceLoc Loc, PresumedLoc PLoc) override;

  void emitImportLocation(FullSourceLoc Loc, PresumedLoc PLoc,
                          StringRef ModuleName) override;

  void emitBuildingModuleLocation(FullSourceLoc Loc, PresumedLoc PLoc,
                                  StringRef ModuleName) override;

private:
  class Node {
  public:
    // Subclasses
    struct Result {
      DiagnosticsEngine::Level Level;
      std::string Message;
      DiagOrStoredDiag Diag;
    };

    struct Location {
      FullSourceLoc Loc;
      PresumedLoc PLoc;
      llvm::SmallVector<CharSourceRange> Ranges;

      // Methods to construct a llvm-style location.
      llvm::SmallVector<CharSourceRange> getCharSourceRangesWithOption(const LangOptions& LangOpts);
    };

    // Constructor
    Node(Result Result_, int Nesting);

    // Operations on building a node-tree.
    // Arguments and results are all in node-style.
    Node &getParent();
    Node &getForkableParent();
    llvm::SmallVector<std::unique_ptr<Node>> &getChildrenPtrs();
    Node &addChildResult(Result);
    Node &addLocation(Location);
    Node &addRelatedLocation(Location);
    template <class Func>
    void recursiveForEach(Func&&);

    // Methods to access underlying data for other llvm-components to read from
    // it. Arguments and results are all in llvm-style.
    unsigned getDiagID();
    DiagnosticsEngine::Level getLevel();
    std::string getDiagnosticMessage();
    llvm::SmallVector<CharSourceRange> getLocations(const LangOptions& LangOpts);
    llvm::SmallVector<CharSourceRange> getRelatedLocations(const LangOptions& LangOpts);
    int getNesting();

  private:
    Result Result_;
    llvm::SmallVector<Location> Locations;
    llvm::SmallVector<Location> RelatedLocations;
    int Nesting;
    Node *ParentPtr = nullptr;
    llvm::SmallVector<std::unique_ptr<Node>> ChildrenPtrs = {};
  };

  Node Root;
  Node *Current = &Root;
  const LangOptions* LangOptsPtr;
  SarifDocumentWriter
      *Writer; // Shared between SARIFDiagnosticPrinter and this renderer.
};

} // end namespace clang

#endif
