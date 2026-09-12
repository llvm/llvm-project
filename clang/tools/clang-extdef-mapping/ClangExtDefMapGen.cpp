//===- ClangExtDefMapGen.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
//
// Clang tool which creates a list of defined functions and the files in which
// they are defined.
//
//===--------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CrossTU/CrossTUDiagnostic.h"
#include "clang/CrossTU/CrossTranslationUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include <optional>
#include <string>

using namespace llvm;
using namespace clang;
using namespace clang::cross_tu;
using namespace clang::tooling;

static cl::OptionCategory
    ClangExtDefMapGenCategory("clang-extdef-mapping options");

class ExtDefNameMap {
  llvm::StringMap<std::string> Index;
  llvm::StringSet<> WeakNames;

public:
  bool addName(const std::string &USR, bool IsWeak,
               const std::string &FileName) {
    bool NameExists = Index.contains(USR);
    if (IsWeak) {
      if (!NameExists) {
        Index[USR] = FileName;
        WeakNames.insert(USR);
      }
    } else {
      bool ExistingIsWeak = WeakNames.erase(USR);
      if (NameExists && !ExistingIsWeak)
        return false;
      Index[USR] = FileName;
    }
    return true;
  }

  const std::string &lookupName(const std::string &USR) const {
    return Index.at(USR);
  }

  void printIndex() const { llvm::outs() << createCrossTUIndexString(Index); }
};

class MapExtDefNamesConsumer : public ASTConsumer {
public:
  MapExtDefNamesConsumer(ExtDefNameMap &NMap, ASTContext &Context,
                         StringRef astFilePath = StringRef())
      : Ctx(Context), SM(Context.getSourceManager()), NameMap(NMap) {
    CurrentFileName = astFilePath.str();
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    handleDecl(Context.getTranslationUnitDecl());
  }

private:
  void handleDecl(const Decl *D);
  void addIfInMain(const DeclaratorDecl *DD, SourceLocation defStart);

  ASTContext &Ctx;
  SourceManager &SM;
  std::string CurrentFileName;
  ExtDefNameMap &NameMap;
};

void MapExtDefNamesConsumer::handleDecl(const Decl *D) {
  if (!D)
    return;

  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isThisDeclarationADefinition())
      if (const Stmt *Body = FD->getBody())
        addIfInMain(FD, Body->getBeginLoc());
  } else if (const auto *VD = dyn_cast<VarDecl>(D)) {
    if (cross_tu::shouldImport(VD, Ctx) && VD->hasInit())
      if (const Expr *Init = VD->getInit())
        addIfInMain(VD, Init->getBeginLoc());
  }

  if (const auto *DC = dyn_cast<DeclContext>(D))
    for (const Decl *D : DC->decls())
      handleDecl(D);
}

void MapExtDefNamesConsumer::addIfInMain(const DeclaratorDecl *DD,
                                         SourceLocation defStart) {
  std::optional<std::string> LookupName =
      CrossTranslationUnitContext::getLookupName(DD);
  if (!LookupName)
    return;
  assert(!LookupName->empty() && "Lookup name should be non-empty.");

  if (CurrentFileName.empty()) {
    CurrentFileName = std::string(
        SM.getFileEntryForID(SM.getMainFileID())->tryGetRealPathName());
    if (CurrentFileName.empty())
      CurrentFileName = "invalid_file";
  }

  switch (DD->getLinkageInternal()) {
  case Linkage::External:
  case Linkage::VisibleNone:
  case Linkage::UniqueExternal:
    if (SM.isInMainFile(defStart)) {
      if (!NameMap.addName(*LookupName, DD->hasAttr<WeakAttr>(),
                           CurrentFileName)) {
        Ctx.getDiagnostics().Report(DD->getLocation(),
                                    diag::warn_multiple_def_index)
            << NameMap.lookupName(*LookupName);
      }
    }
    break;
  case Linkage::Invalid:
    llvm_unreachable("Linkage has not been computed!");
  default:
    break;
  }
}

class MapExtDefNamesAction : public ASTFrontendAction {
public:
  MapExtDefNamesAction(ExtDefNameMap &NMap) : NameMap(NMap) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    return std::make_unique<MapExtDefNamesConsumer>(NameMap,
                                                    CI.getASTContext());
  }

private:
  ExtDefNameMap &NameMap;
};

class MapExtDefNamesActionFactory : public FrontendActionFactory {
public:
  MapExtDefNamesActionFactory(ExtDefNameMap &NMap) : NameMap(NMap) {}
  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<MapExtDefNamesAction>(NameMap);
  };

private:
  ExtDefNameMap &NameMap;
};

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

static IntrusiveRefCntPtr<DiagnosticsEngine> Diags;

static IntrusiveRefCntPtr<DiagnosticsEngine>
GetDiagnosticsEngine(DiagnosticOptions &DiagOpts) {
  if (Diags) {
    // Call reset to make sure we don't mix errors
    Diags->Reset(false);
    return Diags;
  }

  TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(llvm::errs(), DiagOpts);
  DiagClient->setPrefix("clang-extdef-mapping");

  auto DiagEngine = llvm::makeIntrusiveRefCnt<DiagnosticsEngine>(
      DiagnosticIDs::create(), DiagOpts, DiagClient);
  Diags.swap(DiagEngine);

  // Retain this one time so it's not destroyed by ASTUnit::LoadFromASTFile
  Diags->Retain();
  return Diags;
}

static CompilerInstance *CI = nullptr;

static bool HandleAST(ExtDefNameMap &NMap, StringRef AstPath) {

  if (!CI)
    CI = new CompilerInstance();

  auto DiagOpts = std::make_shared<DiagnosticOptions>();
  IntrusiveRefCntPtr<DiagnosticsEngine> DiagEngine =
      GetDiagnosticsEngine(*DiagOpts);

  std::unique_ptr<ASTUnit> Unit = ASTUnit::LoadFromASTFile(
      AstPath, CI->getPCHContainerOperations()->getRawReader(),
      ASTUnit::LoadASTOnly, CI->getVirtualFileSystemPtr(), DiagOpts, DiagEngine,
      CI->getFileSystemOpts(), CI->getHeaderSearchOpts());

  if (!Unit)
    return false;

  FileManager FM(CI->getFileSystemOpts());
  SmallString<128> AbsPath(AstPath);
  FM.makeAbsolutePath(AbsPath);

  MapExtDefNamesConsumer Consumer =
      MapExtDefNamesConsumer(NMap, Unit->getASTContext(), AbsPath);
  Consumer.HandleTranslationUnit(Unit->getASTContext());

  return true;
}

static int HandleFiles(ArrayRef<std::string> SourceFiles,
                       CompilationDatabase &compilations) {
  ExtDefNameMap NameMap;
  std::vector<std::string> SourcesToBeParsed;

  // Loop over all input files, if they are pre-compiled AST
  // process them directly in HandleAST, otherwise put them
  // on a list for ClangTool to handle.
  for (StringRef Src : SourceFiles) {
    if (Src.ends_with(".ast")) {
      if (!HandleAST(NameMap, Src)) {
        return 1;
      }
    } else {
      SourcesToBeParsed.push_back(Src.str());
    }
  }

  MapExtDefNamesActionFactory Factory(NameMap);
  ClangTool Tool(compilations, SourcesToBeParsed);
  int Ret = Tool.run(&Factory);

  NameMap.printIndex();

  return Ret;
}

int main(int argc, const char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0], false);
  PrettyStackTraceProgram X(argc, argv);

  const char *Overview = "\nThis tool collects the USR name and location "
                         "of external definitions in the source files "
                         "(excluding headers).\n"
                         "Input can be either source files that are compiled "
                         "with compile database or .ast files that are "
                         "created from clang's -emit-ast option.\n";
  auto ExpectedParser = CommonOptionsParser::create(
      argc, argv, ClangExtDefMapGenCategory, cl::OneOrMore, Overview);
  if (!ExpectedParser) {
    llvm::WithColor::error() << llvm::toString(ExpectedParser.takeError());
    return 1;
  }
  CommonOptionsParser &OptionsParser = ExpectedParser.get();

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  return HandleFiles(OptionsParser.getSourcePathList(),
                     OptionsParser.getCompilations());
}
