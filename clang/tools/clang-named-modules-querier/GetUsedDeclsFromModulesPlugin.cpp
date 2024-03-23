//===- GetUsedDeclsFromModulesPlugin.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GetDeclsInfoToJson.h"

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Serialization/ASTDeserializationListener.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

namespace clang {

class DeclsQuerier : public ASTDeserializationListener {
public:
  void DeclRead(serialization::DeclID ID, const Decl *D) override {
    if (Stopped)
      return;

    // We only cares about function decls, var decls, tag decls (class, struct, enum, union).
    if (!isa<NamedDecl>(D))
      return;
    
    // Filter Decl's to avoid store too many informations.
    if (!D->getLexicalDeclContext())
      return;

    // We only records the template declaration if the declaration is placed in templates.
    if (auto *FD = dyn_cast<FunctionDecl>(D); FD && FD->getDescribedFunctionTemplate())
      return;

    if (auto *VD = dyn_cast<VarDecl>(D); VD && VD->getDescribedVarTemplate())
      return;

    if (auto *CRD = dyn_cast<CXXRecordDecl>(D); CRD && CRD->getDescribedClassTemplate())
      return;

    if (isa<TemplateTypeParmDecl, NonTypeTemplateParmDecl, TemplateTemplateParmDecl>(D))
      return;
        
    // We don't care about declarations in function scope. 
    if (isa<FunctionDecl>(D->getDeclContext()))
      return;
    
    // Skip implicit declarations.
    if (D->isImplicit())
      return;

    Module *M = D->getOwningModule();
    // We only cares about C++20 Named Modules.
    if (!M || !M->getTopLevelModule()->isNamedModule())
      return;

    StringRef ModuleName = M->Name;
    auto Iter = Names.find(ModuleName);
    if (Iter == Names.end())
      Iter = Names.try_emplace(ModuleName, std::vector<const NamedDecl*>()).first;
    
    Iter->second.push_back(cast<NamedDecl>(D));
  }

  llvm::StringMap<std::vector<const NamedDecl *>> Names;
  bool Stopped = false;
};

class DeclsQuerierConsumer : public ASTConsumer {
  CompilerInstance &CI;
  StringRef InFile;
  std::string OutputFile;
  DeclsQuerier Querier;
  
public:
  DeclsQuerierConsumer(CompilerInstance &CI, StringRef InFile, StringRef OutputFile)
    : CI(CI), InFile(InFile), OutputFile(OutputFile) {}

  ASTDeserializationListener *GetASTDeserializationListener() override {
    return &Querier;
  }

  std::unique_ptr<raw_pwrite_stream> getOutputFile() {
    if (OutputFile.empty()) {
      llvm::SmallString<256> Path(InFile);
      llvm::sys::path::replace_extension(Path, "used_external_decls.json");
      OutputFile = (std::string)Path;
    }

    std::error_code EC;
    auto OS = std::make_unique<llvm::raw_fd_ostream>(OutputFile, EC);
    if (EC)
      return nullptr;
    
    return OS;
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    std::unique_ptr<raw_pwrite_stream> OS = getOutputFile();
    if (!OS)
      return;
    
    /// Otherwise the process of computing ODR Hash may involve more decls
    /// get deserialized.
    Querier.Stopped = true;

    using namespace llvm::json;

    Array Modules;

    for (auto &Iter : Querier.Names) {
      Object ModulesInfo;

      StringRef ModuleName = Iter.first();
      ModulesInfo.try_emplace("module", ModuleName);

      std::vector<const NamedDecl *> Decls(Iter.second);
      Array DeclsInJson;
      for (auto *ND : Decls)
        DeclsInJson.push_back(getDeclInJson(ND, Ctx.getSourceManager()));

      ModulesInfo.try_emplace("decls", std::move(DeclsInJson));
      Modules.push_back(std::move(ModulesInfo));
    }

    *OS << llvm::formatv("{0:2}\n", Value(std::move(Modules)));
  }
};

void PrintHelp();

class DeclsQueryAction : public PluginASTAction {
  std::string OutputFile;

public:
  DeclsQueryAction(StringRef OutputFile) : OutputFile(OutputFile) {}
  DeclsQueryAction() = default;

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<DeclsQuerierConsumer>(CI, InFile, OutputFile);
  }

  ActionType getActionType() override { return AddAfterMainAction; }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &Args) override {
    for (auto &Arg : Args) {
      if (StringRef(Arg).startswith("output=")) {
        OutputFile = StringRef(Arg).split('=').second;
      } else {
        PrintHelp();
        return false;
      }
    }

    return true;
  }
};

void PrintHelp() {
  llvm::outs() << R"cpp(
To get used decls from modules.

The output is printed to the std output by default when use it as a standalone tool.

If you're using plugin, use -fplugin-arg-get_used_decls_from_modules-output=<output-file>
to specify the output path of used decls.
  )cpp";
}
}

static clang::FrontendPluginRegistry::Add<clang::DeclsQueryAction>
X("get_used_decls_from_modules", "query used decls from modules");
