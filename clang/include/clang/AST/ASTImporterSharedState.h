//===- ASTImporterSharedState.h - ASTImporter specific state --*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTImporter specific state, which may be shared
//  amongst several ASTImporter objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTIMPORTERSHAREDSTATE_H
#define LLVM_CLANG_AST_ASTIMPORTERSHAREDSTATE_H

#include "clang/AST/ASTImporterLookupTable.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/DenseMap.h"
// FIXME We need this because of ImportError.
#include "clang/AST/ASTImporter.h"

namespace clang {

class TranslationUnitDecl;

/// Importer specific state, which may be shared amongst several ASTImporter
/// objects.
class ASTImporterSharedState {

  /// Pointer to the import specific lookup table.
  std::unique_ptr<ASTImporterLookupTable> LookupTable;

  /// Mapping from the already-imported declarations in the "to"
  /// context to the error status of the import of that declaration.
  /// This map contains only the declarations that were not correctly
  /// imported. The same declaration may or may not be included in
  /// ImportedFromDecls. This map is updated continuously during imports and
  /// never cleared (like ImportedFromDecls).
  llvm::DenseMap<Decl *, ImportError> ImportErrors;

  // FIXME put ImportedFromDecls here!
  // And from that point we can better encapsulate the lookup table.

public:
  ASTImporterSharedState() = default;

  ASTImporterSharedState(TranslationUnitDecl &ToTU) {
    LookupTable = llvm::make_unique<ASTImporterLookupTable>(ToTU);
  }

  ASTImporterLookupTable *getLookupTable() { return LookupTable.get(); }

  void addDeclToLookup(Decl *D) {
    if (LookupTable)
      if (auto *ND = dyn_cast<NamedDecl>(D))
        LookupTable->add(ND);
  }

  void removeDeclFromLookup(Decl *D) {
    if (LookupTable)
      if (auto *ND = dyn_cast<NamedDecl>(D))
        LookupTable->remove(ND);
  }

  llvm::Optional<ImportError> getImportDeclErrorIfAny(Decl *ToD) const {
    auto Pos = ImportErrors.find(ToD);
    if (Pos != ImportErrors.end())
      return Pos->second;
    else
      return Optional<ImportError>();
  }

  void setImportDeclError(Decl *To, ImportError Error) {
    ImportErrors[To] = Error;
  }
};

} // namespace clang
#endif // LLVM_CLANG_AST_ASTIMPORTERSHAREDSTATE_H
