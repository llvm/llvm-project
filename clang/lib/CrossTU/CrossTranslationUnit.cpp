//===--- CrossTranslationUnit.cpp - -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the CrossTranslationUnit interface.
//
//===----------------------------------------------------------------------===//
#include "clang/CrossTU/CrossTranslationUnit.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CrossTU/CrossTUDiagnostic.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <sstream>

namespace clang {
namespace cross_tu {

namespace {
// FIXME: This class is will be removed after the transition to llvm::Error.
class IndexErrorCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "clang.index"; }

  std::string message(int Condition) const override {
    switch (static_cast<index_error_code>(Condition)) {
    case index_error_code::unspecified:
      return "An unknown error has occurred.";
    case index_error_code::missing_index_file:
      return "The index file is missing.";
    case index_error_code::invalid_index_format:
      return "Invalid index file format.";
    case index_error_code::multiple_definitions:
      return "Multiple definitions in the index file.";
    case index_error_code::missing_definition:
      return "Missing definition from the index file.";
    case index_error_code::failed_import:
      return "Failed to import the definition.";
    case index_error_code::failed_to_get_external_ast:
      return "Failed to load external AST source.";
    case index_error_code::failed_to_generate_usr:
      return "Failed to generate USR.";
    }
    llvm_unreachable("Unrecognized index_error_code.");
  }
};

static llvm::ManagedStatic<IndexErrorCategory> Category;
} // end anonymous namespace

char IndexError::ID;

void IndexError::log(raw_ostream &OS) const {
  OS << Category->message(static_cast<int>(Code)) << '\n';
}

std::error_code IndexError::convertToErrorCode() const {
  return std::error_code(static_cast<int>(Code), *Category);
}

llvm::Expected<llvm::StringMap<std::string>>
parseCrossTUIndex(StringRef IndexPath, StringRef CrossTUDir) {
  std::ifstream ExternalFnMapFile(IndexPath);
  if (!ExternalFnMapFile)
    return llvm::make_error<IndexError>(index_error_code::missing_index_file,
                                        IndexPath.str());

  llvm::StringMap<std::string> Result;
  std::string Line;
  unsigned LineNo = 1;
  while (std::getline(ExternalFnMapFile, Line)) {
    const size_t Pos = Line.find(" ");
    if (Pos > 0 && Pos != std::string::npos) {
      StringRef LineRef{Line};
      StringRef FunctionLookupName = LineRef.substr(0, Pos);
      if (Result.count(FunctionLookupName))
        return llvm::make_error<IndexError>(
            index_error_code::multiple_definitions, IndexPath.str(), LineNo);
      StringRef FileName = LineRef.substr(Pos + 1);
      SmallString<256> FilePath = CrossTUDir;
      llvm::sys::path::append(FilePath, FileName);
      Result[FunctionLookupName] = FilePath.str().str();
    } else
      return llvm::make_error<IndexError>(
          index_error_code::invalid_index_format, IndexPath.str(), LineNo);
    LineNo++;
  }
  return Result;
}

std::string
createCrossTUIndexString(const llvm::StringMap<std::string> &Index) {
  std::ostringstream Result;
  for (const auto &E : Index)
    Result << E.getKey().str() << " " << E.getValue() << '\n';
  return Result.str();
}

CrossTranslationUnitContext::CrossTranslationUnitContext(CompilerInstance &CI)
    : CI(CI), Context(CI.getASTContext()) {}

CrossTranslationUnitContext::~CrossTranslationUnitContext() {}

std::string CrossTranslationUnitContext::getLookupName(const NamedDecl *ND) {
  SmallString<128> DeclUSR;
  bool Ret = index::generateUSRForDecl(ND, DeclUSR); (void)Ret;
  assert(!Ret && "Unable to generate USR");
  return DeclUSR.str();
}

/// Recursively visits the function decls of a DeclContext, and looks up a
/// function based on USRs.
const FunctionDecl *
CrossTranslationUnitContext::findFunctionInDeclContext(const DeclContext *DC,
                                                       StringRef LookupFnName) {
  assert(DC && "Declaration Context must not be null");
  for (const Decl *D : DC->decls()) {
    const auto *SubDC = dyn_cast<DeclContext>(D);
    if (SubDC)
      if (const auto *FD = findFunctionInDeclContext(SubDC, LookupFnName))
        return FD;

    const auto *ND = dyn_cast<FunctionDecl>(D);
    const FunctionDecl *ResultDecl;
    if (!ND || !ND->hasBody(ResultDecl))
      continue;
    if (getLookupName(ResultDecl) != LookupFnName)
      continue;
    return ResultDecl;
  }
  return nullptr;
}

llvm::Expected<const FunctionDecl *>
CrossTranslationUnitContext::getCrossTUDefinition(const FunctionDecl *FD,
                                                  StringRef CrossTUDir,
                                                  StringRef IndexName) {
  assert(!FD->hasBody() && "FD has a definition in current translation unit!");
  const std::string LookupFnName = getLookupName(FD);
  if (LookupFnName.empty())
    return llvm::make_error<IndexError>(
        index_error_code::failed_to_generate_usr);
  llvm::Expected<ASTUnit *> ASTUnitOrError =
      loadExternalAST(LookupFnName, CrossTUDir, IndexName);
  if (!ASTUnitOrError)
    return ASTUnitOrError.takeError();
  ASTUnit *Unit = *ASTUnitOrError;
  if (!Unit)
    return llvm::make_error<IndexError>(
        index_error_code::failed_to_get_external_ast);
  assert(&Unit->getFileManager() ==
         &Unit->getASTContext().getSourceManager().getFileManager());

  TranslationUnitDecl *TU = Unit->getASTContext().getTranslationUnitDecl();
  if (const FunctionDecl *ResultDecl =
          findFunctionInDeclContext(TU, LookupFnName))
    return importDefinition(ResultDecl);
  return llvm::make_error<IndexError>(index_error_code::failed_import);
}

void CrossTranslationUnitContext::emitCrossTUDiagnostics(const IndexError &IE) {
  switch (IE.getCode()) {
  case index_error_code::missing_index_file:
    Context.getDiagnostics().Report(diag::err_fe_error_opening)
        << IE.getFileName() << "required by the CrossTU functionality";
    break;
  case index_error_code::invalid_index_format:
    Context.getDiagnostics().Report(diag::err_fnmap_parsing)
        << IE.getFileName() << IE.getLineNum();
    break;
  case index_error_code::multiple_definitions:
    Context.getDiagnostics().Report(diag::err_multiple_def_index)
        << IE.getLineNum();
    break;
  default:
    break;
  }
}

llvm::Expected<ASTUnit *> CrossTranslationUnitContext::loadExternalAST(
    StringRef LookupName, StringRef CrossTUDir, StringRef IndexName) {
  // FIXME: The current implementation only supports loading functions with
  //        a lookup name from a single translation unit. If multiple
  //        translation units contains functions with the same lookup name an
  //        error will be returned.
  ASTUnit *Unit = nullptr;
  auto FnUnitCacheEntry = FunctionASTUnitMap.find(LookupName);
  if (FnUnitCacheEntry == FunctionASTUnitMap.end()) {
    if (FunctionFileMap.empty()) {
      SmallString<256> IndexFile = CrossTUDir;
      if (llvm::sys::path::is_absolute(IndexName))
        IndexFile = IndexName;
      else
        llvm::sys::path::append(IndexFile, IndexName);
      llvm::Expected<llvm::StringMap<std::string>> IndexOrErr =
          parseCrossTUIndex(IndexFile, CrossTUDir);
      if (IndexOrErr)
        FunctionFileMap = *IndexOrErr;
      else
        return IndexOrErr.takeError();
    }

    auto It = FunctionFileMap.find(LookupName);
    if (It == FunctionFileMap.end())
      return llvm::make_error<IndexError>(index_error_code::missing_definition);
    StringRef ASTFileName = It->second;
    auto ASTCacheEntry = FileASTUnitMap.find(ASTFileName);
    if (ASTCacheEntry == FileASTUnitMap.end()) {
      IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
      TextDiagnosticPrinter *DiagClient =
          new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);
      IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
      IntrusiveRefCntPtr<DiagnosticsEngine> Diags(
          new DiagnosticsEngine(DiagID, &*DiagOpts, DiagClient));

      std::unique_ptr<ASTUnit> LoadedUnit(ASTUnit::LoadFromASTFile(
          ASTFileName, CI.getPCHContainerOperations()->getRawReader(),
          ASTUnit::LoadEverything, Diags, CI.getFileSystemOpts()));
      Unit = LoadedUnit.get();
      FileASTUnitMap[ASTFileName] = std::move(LoadedUnit);
    } else {
      Unit = ASTCacheEntry->second.get();
    }
    FunctionASTUnitMap[LookupName] = Unit;
  } else {
    Unit = FnUnitCacheEntry->second;
  }
  return Unit;
}

llvm::Expected<const FunctionDecl *>
CrossTranslationUnitContext::importDefinition(const FunctionDecl *FD) {
  ASTImporter &Importer = getOrCreateASTImporter(FD->getASTContext());
  auto *ToDecl =
      cast<FunctionDecl>(Importer.Import(const_cast<FunctionDecl *>(FD)));
  assert(ToDecl->hasBody());
  assert(FD->hasBody() && "Functions already imported should have body.");
  return ToDecl;
}

ASTImporter &
CrossTranslationUnitContext::getOrCreateASTImporter(ASTContext &From) {
  auto I = ASTUnitImporterMap.find(From.getTranslationUnitDecl());
  if (I != ASTUnitImporterMap.end())
    return *I->second;
  ASTImporter *NewImporter =
      new ASTImporter(Context, Context.getSourceManager().getFileManager(),
                      From, From.getSourceManager().getFileManager(), false);
  ASTUnitImporterMap[From.getTranslationUnitDecl()].reset(NewImporter);
  return *NewImporter;
}

} // namespace cross_tu
} // namespace clang
