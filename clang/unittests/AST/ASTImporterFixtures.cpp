//===- unittest/AST/ASTImporterFixtures.cpp - AST unit test support -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implementation of fixture classes for testing the ASTImporter.
//
//===----------------------------------------------------------------------===//

#include "ASTImporterFixtures.h"

#include "clang/AST/ASTImporter.h"
#include "clang/AST/ASTImporterLookupTable.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"

namespace clang {
namespace ast_matchers {

void createVirtualFileIfNeeded(ASTUnit *ToAST, StringRef FileName,
                               std::unique_ptr<llvm::MemoryBuffer> &&Buffer) {
  assert(ToAST);
  ASTContext &ToCtx = ToAST->getASTContext();
  auto *OFS = static_cast<llvm::vfs::OverlayFileSystem *>(
      &ToCtx.getSourceManager().getFileManager().getVirtualFileSystem());
  auto *MFS = static_cast<llvm::vfs::InMemoryFileSystem *>(
      OFS->overlays_begin()->get());
  MFS->addFile(FileName, 0, std::move(Buffer));
}

void createVirtualFileIfNeeded(ASTUnit *ToAST, StringRef FileName,
                               StringRef Code) {
  return createVirtualFileIfNeeded(ToAST, FileName,
                                   llvm::MemoryBuffer::getMemBuffer(Code));
}

ASTImporterTestBase::TU::TU(StringRef Code, StringRef FileName, ArgVector Args,
                            ImporterConstructor C)
    : Code(Code), FileName(FileName),
      Unit(tooling::buildASTFromCodeWithArgs(this->Code, Args, this->FileName)),
      TUDecl(Unit->getASTContext().getTranslationUnitDecl()), Creator(C) {
  Unit->enableSourceFileDiagnostics();

  // If the test doesn't need a specific ASTImporter, we just create a
  // normal ASTImporter with it.
  if (!Creator)
    Creator = [](ASTContext &ToContext, FileManager &ToFileManager,
                 ASTContext &FromContext, FileManager &FromFileManager,
                 bool MinimalImport, ASTImporterLookupTable *LookupTable) {
      return new ASTImporter(ToContext, ToFileManager, FromContext,
                             FromFileManager, MinimalImport, LookupTable);
    };
}

ASTImporterTestBase::TU::~TU() {}

void ASTImporterTestBase::TU::lazyInitImporter(
    ASTImporterLookupTable &LookupTable, ASTUnit *ToAST) {
  assert(ToAST);
  if (!Importer)
    Importer.reset(Creator(ToAST->getASTContext(), ToAST->getFileManager(),
                           Unit->getASTContext(), Unit->getFileManager(), false,
                           &LookupTable));
  assert(&ToAST->getASTContext() == &Importer->getToContext());
  createVirtualFileIfNeeded(ToAST, FileName, Code);
}

Decl *ASTImporterTestBase::TU::import(ASTImporterLookupTable &LookupTable,
                                      ASTUnit *ToAST, Decl *FromDecl) {
  lazyInitImporter(LookupTable, ToAST);
  if (auto ImportedOrErr = Importer->Import(FromDecl))
    return *ImportedOrErr;
  else {
    llvm::consumeError(ImportedOrErr.takeError());
    return nullptr;
  }
}

QualType ASTImporterTestBase::TU::import(ASTImporterLookupTable &LookupTable,
                                         ASTUnit *ToAST, QualType FromType) {
  lazyInitImporter(LookupTable, ToAST);
  if (auto ImportedOrErr = Importer->Import(FromType))
    return *ImportedOrErr;
  else {
    llvm::consumeError(ImportedOrErr.takeError());
    return QualType{};
  }
}

void ASTImporterTestBase::lazyInitLookupTable(TranslationUnitDecl *ToTU) {
  assert(ToTU);
  if (!LookupTablePtr)
    LookupTablePtr = llvm::make_unique<ASTImporterLookupTable>(*ToTU);
}

void ASTImporterTestBase::lazyInitToAST(Language ToLang, StringRef ToSrcCode,
                                        StringRef FileName) {
  if (ToAST)
    return;
  ArgVector ToArgs = getArgVectorForLanguage(ToLang);
  // Source code must be a valid live buffer through the tests lifetime.
  ToCode = ToSrcCode;
  // Build the AST from an empty file.
  ToAST = tooling::buildASTFromCodeWithArgs(ToCode, ToArgs, FileName);
  ToAST->enableSourceFileDiagnostics();
  lazyInitLookupTable(ToAST->getASTContext().getTranslationUnitDecl());
}

ASTImporterTestBase::TU *ASTImporterTestBase::findFromTU(Decl *From) {
  // Create a virtual file in the To Ctx which corresponds to the file from
  // which we want to import the `From` Decl. Without this source locations
  // will be invalid in the ToCtx.
  auto It = llvm::find_if(FromTUs, [From](const TU &E) {
    return E.TUDecl == From->getTranslationUnitDecl();
  });
  assert(It != FromTUs.end());
  return &*It;
}

std::tuple<Decl *, Decl *>
ASTImporterTestBase::getImportedDecl(StringRef FromSrcCode, Language FromLang,
                                     StringRef ToSrcCode, Language ToLang,
                                     StringRef Identifier) {
  ArgVector FromArgs = getArgVectorForLanguage(FromLang),
            ToArgs = getArgVectorForLanguage(ToLang);

  FromTUs.emplace_back(FromSrcCode, InputFileName, FromArgs, Creator);
  TU &FromTU = FromTUs.back();

  assert(!ToAST);
  lazyInitToAST(ToLang, ToSrcCode, OutputFileName);

  ASTContext &FromCtx = FromTU.Unit->getASTContext();

  IdentifierInfo *ImportedII = &FromCtx.Idents.get(Identifier);
  assert(ImportedII && "Declaration with the given identifier "
                       "should be specified in test!");
  DeclarationName ImportDeclName(ImportedII);
  SmallVector<NamedDecl *, 1> FoundDecls;
  FromCtx.getTranslationUnitDecl()->localUncachedLookup(ImportDeclName,
                                                        FoundDecls);

  assert(FoundDecls.size() == 1);

  Decl *Imported =
      FromTU.import(*LookupTablePtr, ToAST.get(), FoundDecls.front());

  assert(Imported);
  return std::make_tuple(*FoundDecls.begin(), Imported);
}

TranslationUnitDecl *ASTImporterTestBase::getTuDecl(StringRef SrcCode,
                                                    Language Lang,
                                                    StringRef FileName) {
  assert(llvm::find_if(FromTUs, [FileName](const TU &E) {
           return E.FileName == FileName;
         }) == FromTUs.end());

  ArgVector Args = getArgVectorForLanguage(Lang);
  FromTUs.emplace_back(SrcCode, FileName, Args);
  TU &Tu = FromTUs.back();

  return Tu.TUDecl;
}

TranslationUnitDecl *ASTImporterTestBase::getToTuDecl(StringRef ToSrcCode,
                                                      Language ToLang) {
  ArgVector ToArgs = getArgVectorForLanguage(ToLang);
  assert(!ToAST);
  lazyInitToAST(ToLang, ToSrcCode, OutputFileName);
  return ToAST->getASTContext().getTranslationUnitDecl();
}

Decl *ASTImporterTestBase::Import(Decl *From, Language ToLang) {
  lazyInitToAST(ToLang, "", OutputFileName);
  TU *FromTU = findFromTU(From);
  assert(LookupTablePtr);
  return FromTU->import(*LookupTablePtr, ToAST.get(), From);
}

QualType ASTImporterTestBase::ImportType(QualType FromType, Decl *TUDecl,
                                         Language ToLang) {
  lazyInitToAST(ToLang, "", OutputFileName);
  TU *FromTU = findFromTU(TUDecl);
  assert(LookupTablePtr);
  return FromTU->import(*LookupTablePtr, ToAST.get(), FromType);
}

ASTImporterTestBase::~ASTImporterTestBase() {
  if (!::testing::Test::HasFailure())
    return;

  for (auto &Tu : FromTUs) {
    assert(Tu.Unit);
    llvm::errs() << "FromAST:\n";
    Tu.Unit->getASTContext().getTranslationUnitDecl()->dump();
    llvm::errs() << "\n";
  }
  if (ToAST) {
    llvm::errs() << "ToAST:\n";
    ToAST->getASTContext().getTranslationUnitDecl()->dump();
  }
}

} // end namespace ast_matchers
} // end namespace clang
