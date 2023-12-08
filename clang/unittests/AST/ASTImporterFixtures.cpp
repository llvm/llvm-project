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
#include "clang/AST/ASTImporterSharedState.h"
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

ASTImporterTestBase::TU::TU(StringRef Code, StringRef FileName,
                            std::vector<std::string> Args,
                            ImporterConstructor C,
                            ASTImporter::ODRHandlingType ODRHandling)
    : Code(std::string(Code)), FileName(std::string(FileName)),
      Unit(tooling::buildASTFromCodeWithArgs(this->Code, Args, this->FileName)),
      TUDecl(Unit->getASTContext().getTranslationUnitDecl()), Creator(C),
      ODRHandling(ODRHandling) {
  Unit->enableSourceFileDiagnostics();

  // If the test doesn't need a specific ASTImporter, we just create a
  // normal ASTImporter with it.
  if (!Creator)
    Creator = [](ASTContext &ToContext, FileManager &ToFileManager,
                 ASTContext &FromContext, FileManager &FromFileManager,
                 bool MinimalImport,
                 const std::shared_ptr<ASTImporterSharedState> &SharedState) {
      return new ASTImporter(ToContext, ToFileManager, FromContext,
                             FromFileManager, MinimalImport, SharedState);
    };
}

ASTImporterTestBase::TU::~TU() {}

void ASTImporterTestBase::TU::lazyInitImporter(
    const std::shared_ptr<ASTImporterSharedState> &SharedState,
    ASTUnit *ToAST) {
  assert(ToAST);
  if (!Importer) {
    Importer.reset(Creator(ToAST->getASTContext(), ToAST->getFileManager(),
                           Unit->getASTContext(), Unit->getFileManager(), false,
                           SharedState));
    Importer->setODRHandling(ODRHandling);
  }
  assert(&ToAST->getASTContext() == &Importer->getToContext());
  createVirtualFileIfNeeded(ToAST, FileName, Code);
}

Decl *ASTImporterTestBase::TU::import(
    const std::shared_ptr<ASTImporterSharedState> &SharedState, ASTUnit *ToAST,
    Decl *FromDecl) {
  lazyInitImporter(SharedState, ToAST);
  if (auto ImportedOrErr = Importer->Import(FromDecl))
    return *ImportedOrErr;
  else {
    llvm::consumeError(ImportedOrErr.takeError());
    return nullptr;
  }
}

llvm::Expected<Decl *> ASTImporterTestBase::TU::importOrError(
    const std::shared_ptr<ASTImporterSharedState> &SharedState, ASTUnit *ToAST,
    Decl *FromDecl) {
  lazyInitImporter(SharedState, ToAST);
  return Importer->Import(FromDecl);
}

QualType ASTImporterTestBase::TU::import(
    const std::shared_ptr<ASTImporterSharedState> &SharedState, ASTUnit *ToAST,
    QualType FromType) {
  lazyInitImporter(SharedState, ToAST);
  if (auto ImportedOrErr = Importer->Import(FromType))
    return *ImportedOrErr;
  else {
    llvm::consumeError(ImportedOrErr.takeError());
    return QualType{};
  }
}

void ASTImporterTestBase::lazyInitSharedState(TranslationUnitDecl *ToTU) {
  assert(ToTU);
  if (!SharedStatePtr)
    SharedStatePtr = std::make_shared<ASTImporterSharedState>(*ToTU);
}

void ASTImporterTestBase::lazyInitToAST(TestLanguage ToLang,
                                        StringRef ToSrcCode,
                                        StringRef FileName) {
  if (ToAST)
    return;
  std::vector<std::string> ToArgs = getCommandLineArgsForLanguage(ToLang);
  // Source code must be a valid live buffer through the tests lifetime.
  ToCode = std::string(ToSrcCode);
  // Build the AST from an empty file.
  ToAST = tooling::buildASTFromCodeWithArgs(ToCode, ToArgs, FileName);
  ToAST->enableSourceFileDiagnostics();
  lazyInitSharedState(ToAST->getASTContext().getTranslationUnitDecl());
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

std::tuple<Decl *, Decl *> ASTImporterTestBase::getImportedDecl(
    StringRef FromSrcCode, TestLanguage FromLang, StringRef ToSrcCode,
    TestLanguage ToLang, StringRef Identifier) {
  std::vector<std::string> FromArgs = getCommandLineArgsForLanguage(FromLang);
  std::vector<std::string> ToArgs = getCommandLineArgsForLanguage(ToLang);

  FromTUs.emplace_back(FromSrcCode, InputFileName, FromArgs, Creator,
                       ODRHandling);
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
      FromTU.import(SharedStatePtr, ToAST.get(), FoundDecls.front());

  assert(Imported);
  return std::make_tuple(*FoundDecls.begin(), Imported);
}

TranslationUnitDecl *ASTImporterTestBase::getTuDecl(StringRef SrcCode,
                                                    TestLanguage Lang,
                                                    StringRef FileName) {
  assert(llvm::find_if(FromTUs, [FileName](const TU &E) {
           return E.FileName == FileName;
         }) == FromTUs.end());

  std::vector<std::string> Args = getCommandLineArgsForLanguage(Lang);
  FromTUs.emplace_back(SrcCode, FileName, Args, Creator, ODRHandling);
  TU &Tu = FromTUs.back();

  return Tu.TUDecl;
}

TranslationUnitDecl *ASTImporterTestBase::getToTuDecl(StringRef ToSrcCode,
                                                      TestLanguage ToLang) {
  std::vector<std::string> ToArgs = getCommandLineArgsForLanguage(ToLang);
  assert(!ToAST);
  lazyInitToAST(ToLang, ToSrcCode, OutputFileName);
  return ToAST->getASTContext().getTranslationUnitDecl();
}

Decl *ASTImporterTestBase::Import(Decl *From, TestLanguage ToLang) {
  lazyInitToAST(ToLang, "", OutputFileName);
  TU *FromTU = findFromTU(From);
  assert(SharedStatePtr);
  Decl *To = FromTU->import(SharedStatePtr, ToAST.get(), From);
  return To;
}

llvm::Expected<Decl *> ASTImporterTestBase::importOrError(Decl *From,
                                                          TestLanguage ToLang) {
  lazyInitToAST(ToLang, "", OutputFileName);
  TU *FromTU = findFromTU(From);
  assert(SharedStatePtr);
  llvm::Expected<Decl *> To =
      FromTU->importOrError(SharedStatePtr, ToAST.get(), From);
  return To;
}

QualType ASTImporterTestBase::ImportType(QualType FromType, Decl *TUDecl,
                                         TestLanguage ToLang) {
  lazyInitToAST(ToLang, "", OutputFileName);
  TU *FromTU = findFromTU(TUDecl);
  assert(SharedStatePtr);
  return FromTU->import(SharedStatePtr, ToAST.get(), FromType);
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
