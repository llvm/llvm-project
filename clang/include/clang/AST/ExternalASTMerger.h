//===--- ExternalASTMerger.h - Merging External AST Interface ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file declares the ExternalASTMerger, which vends a combination of ASTs
//  from several different ASTContext/FileManager pairs
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_EXTERNALASTMERGER_H
#define LLVM_CLANG_AST_EXTERNALASTMERGER_H

#include "clang/AST/ASTImporter.h"
#include "clang/AST/ExternalASTSource.h"

namespace clang {

class ExternalASTMerger : public ExternalASTSource {
public:
  struct ImporterPair {
    std::unique_ptr<ASTImporter> Forward;
    std::unique_ptr<ASTImporter> Reverse;
  };

private:
  std::vector<ImporterPair> Importers;

public:
  struct ImporterEndpoint {
    ASTContext &AST;
    FileManager &FM;
  };
  ExternalASTMerger(const ImporterEndpoint &Target,
                    llvm::ArrayRef<ImporterEndpoint> Sources);

  bool FindExternalVisibleDeclsByName(const DeclContext *DC,
                                      DeclarationName Name) override;

  void
  FindExternalLexicalDecls(const DeclContext *DC,
                           llvm::function_ref<bool(Decl::Kind)> IsKindWeWant,
                           SmallVectorImpl<Decl *> &Result) override;
};

} // end namespace clang

#endif
