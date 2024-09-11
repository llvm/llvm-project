//===- ClangNamedModulesQuerier.cppm --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GetDeclsInfoToJson.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "clang/Serialization/ASTReader.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/JSON.h"

using namespace clang;

class DeclsQueryAction : public ASTFrontendAction {
  std::vector<std::string> QueryingDeclNames;
  llvm::json::Array JsonOutput;

public:
  DeclsQueryAction(std::vector<std::string> &&QueryingDeclNames) :
    QueryingDeclNames(QueryingDeclNames) {} 

  bool BeginInvocation(CompilerInstance &CI) override {
    CI.getHeaderSearchOpts().ModuleFormat = "raw";
    return true;
  }

  DeclContext *getDeclContextByName(ASTReader *Reader, StringRef Name);
  std::optional<SmallVector<NamedDecl *>> getDeclsByName(ASTReader *Reader, StringRef Name);

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<ASTConsumer>();
  }

  void QueryDecls(ASTReader *Reader, StringRef Name);

  void ExecuteAction() override {
    assert(isCurrentFileAST() && "dumping non-AST?");

    ASTReader *Reader = getCurrentASTUnit().getASTReader().get();
    serialization::ModuleFile &MF = Reader->getModuleManager().getPrimaryModule();
    if (!MF.StandardCXXModule) {
      llvm::errs() << "We should only consider standard C++20 Modules.\n";
      return;
    }

    for (auto &Name : QueryingDeclNames)
      QueryDecls(Reader, Name);

    CompilerInstance &CI = getCompilerInstance();
    std::unique_ptr<raw_pwrite_stream> OS = CI.createDefaultOutputFile(/*Binary=*/false);
    if (!OS) {
      llvm::errs() << "Failed to create output file\n";
      return;
    }

    using namespace llvm::json;
    *OS << llvm::formatv("{0:2}\n", Value(std::move(JsonOutput)));
  }
};

static DeclContext *getDeclContext(NamedDecl *ND) {
  if (auto *CTD = dyn_cast<ClassTemplateDecl>(ND))
    return CTD->getTemplatedDecl();

  return dyn_cast<DeclContext>(ND);
}

static DeclContext *getDeclContextFor(const SmallVector<NamedDecl *> &DCCandidates) {
  DeclContext *Result = nullptr;

  for (auto *ND : DCCandidates) {
    auto *DC = getDeclContext(ND);
    if (!DC)
      continue;

    if (!Result)
      Result = DC->getPrimaryContext();
    else if (Result == DC->getPrimaryContext())
      continue;
    else {
      llvm::errs() << "Found multiple decl context: \n";
      cast<Decl>(Result)->dump();
      cast<Decl>(DC)->dump();
    }
  }

  return Result;
}

DeclContext *DeclsQueryAction::getDeclContextByName(ASTReader *Reader, StringRef Name) {
  if (Name.empty())
    return Reader->getContext().getTranslationUnitDecl();

  std::optional<SmallVector<NamedDecl *>> DCCandidates = getDeclsByName(Reader, Name);
  if (!DCCandidates || DCCandidates->empty())
    return nullptr;

  return getDeclContextFor(*DCCandidates);
}

std::optional<SmallVector<NamedDecl *>> DeclsQueryAction::getDeclsByName(ASTReader *Reader, StringRef Name) {
  if (Name.endswith("::"))
    return std::nullopt;

  auto [ParentName, UnqualifiedName] = Name.rsplit("::");

  // This implies that "::" is not in the Name.
  if (ParentName == Name) {
    UnqualifiedName = Name;
    ParentName = StringRef();
  }

  DeclContext *DC = getDeclContextByName(Reader, ParentName);
  if (!DC)
    return std::nullopt;

  IdentifierInfo *II = Reader->get(UnqualifiedName);

  if (!II)
    return std::nullopt;

  llvm::SmallVector<NamedDecl *> Decls;
  Reader->FindVisibleDeclsByName(DC, DeclarationName(II), Decls);

  // TODO: Should we filter here?
  return Decls;
}

void DeclsQueryAction::QueryDecls(ASTReader *Reader, StringRef Name) {
  using namespace llvm::json;

  std::optional<SmallVector<NamedDecl *>> Decls = getDeclsByName(Reader, Name);
  if (!Decls) {
    JsonOutput.push_back(Object{{Name, "invalid name"}});
    return;
  }

  SourceManager &SMgr = Reader->getSourceManager();

  // TODO: Handle overloads here.
  for (NamedDecl *ND : *Decls)
    JsonOutput.push_back(getDeclInJson(ND, SMgr));
}

// TODO: Print --help information
// TODO: Add -resource-dir automatically
int clang_named_modules_querier_main(int argc, char **argv, const llvm::ToolContext &) {
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
    CompilerInstance::createDiagnostics(new DiagnosticOptions());
  CreateInvocationOptions CIOpts;
  CIOpts.Diags = Diags;
  CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();

  llvm::ArrayRef<const char *> Args(argv, argv + argc);
  if (llvm::find_if(Args, [](auto &&Arg) {
    return std::strcmp(Arg, "--help") == 0;
  }) != Args.end()) {
    llvm::outs() << R"cpp(
To query the decls from module files.

Syntax:

  clang-named-modules-querier module-file - <decl-names-to-be-queried>...

For example:

  clang-named-modules-querier a.pcm -- a nn::a Templ::get
  
The unqualified name are treated as if it is under the global namespace.

The output information contains kind of the declaration, source file name,
line and col number and the hash value of declaration.
    )cpp";
    return 0;
  }

  auto DashDashIter = llvm::find_if(Args, [](auto &&V){
    return std::strcmp(V, "--") == 0;
  });
  
  std::vector<std::string> QueryingDeclNames;
  auto Iter = DashDashIter;
  // Don't record "--".
  if (Iter != Args.end())
    Iter++;
  while (Iter != Args.end())
    QueryingDeclNames.push_back(std::string(*Iter++));

  if (QueryingDeclNames.empty()) {
    llvm::errs() << "We need pass the names that need to be queried after '--'";
    return 0;
  }

  std::shared_ptr<CompilerInvocation> Invocation =
    createInvocation(llvm::ArrayRef<const char *>(argv, DashDashIter), CIOpts);

  CompilerInstance Instance;
  Instance.setDiagnostics(Diags.get());
  Instance.setInvocation(Invocation);
  DeclsQueryAction Action(std::move(QueryingDeclNames));
  Instance.ExecuteAction(Action);

  return 0;
}
