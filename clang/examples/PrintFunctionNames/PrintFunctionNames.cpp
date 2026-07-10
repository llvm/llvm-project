//===- PrintFunctionNames.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Example clang plugin which simply prints the names of all the top-level decls
// in the input file.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

namespace {

class PrintFunctionsConsumer : public ASTConsumer {
  CompilerInstance &Instance;
  std::set<std::string> ParsedTemplates;
  // Diagnostics in the plugin's own "print-fns-plugin" group, or 0 if the
  // corresponding argument was not passed. Registering the IDs up front (rather
  // than lazily on first use) makes them members of the group before the source
  // is parsed, so a `#pragma clang diagnostic` referring to the group can be
  // applied to them.
  unsigned WarnID = 0;
  unsigned RemarkID = 0;
  unsigned ErrorID = 0;

public:
  PrintFunctionsConsumer(CompilerInstance &Instance,
                         std::set<std::string> ParsedTemplates,
                         bool WarnOnDecls, bool RemarkOnDecls,
                         bool ErrorOnDecls)
      : Instance(Instance), ParsedTemplates(ParsedTemplates) {
    DiagnosticsEngine &Diags = Instance.getDiagnostics();
    // The plugin is registered under "print-fns", so its group is
    // "print-fns-plugin"; passing the plugin's own name keeps the diagnostics
    // in that namespace automatically.
    if (RemarkOnDecls)
      RemarkID = Diags.getCustomPluginDiagID(DiagnosticsEngine::Remark,
                                             "saw top-level declaration '%0'",
                                             "print-fns");
    if (WarnOnDecls)
      WarnID = Diags.getCustomPluginDiagID(
          DiagnosticsEngine::Warning, "suspicious top-level declaration '%0'",
          "print-fns");
    if (ErrorOnDecls)
      ErrorID = Diags.getCustomPluginDiagID(
          DiagnosticsEngine::Error, "forbidden top-level declaration '%0'",
          "print-fns");
  }

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
      const Decl *D = *i;
      const NamedDecl *ND = dyn_cast<NamedDecl>(D);
      if (!ND)
        continue;
      llvm::errs() << "top-level-decl: \"" << ND->getNameAsString() << "\"\n";
      // A user controls the warning with -Wno-print-fns-plugin and the remark
      // with -Rno-print-fns-plugin (or the -Wplugin / -Wno-plugin umbrella),
      // while the error keeps its severity -- group flags never silence errors.
      DiagnosticsEngine &Diags = Instance.getDiagnostics();
      if (RemarkID)
        Diags.Report(ND->getLocation(), RemarkID) << ND->getNameAsString();
      if (WarnID)
        Diags.Report(ND->getLocation(), WarnID) << ND->getNameAsString();
      if (ErrorID)
        Diags.Report(ND->getLocation(), ErrorID) << ND->getNameAsString();
    }

    return true;
  }

  void HandleTranslationUnit(ASTContext& context) override {
    if (!Instance.getLangOpts().DelayedTemplateParsing)
      return;

    // This demonstrates how to force instantiation of some templates in
    // -fdelayed-template-parsing mode. (Note: Doing this unconditionally for
    // all templates is similar to not using -fdelayed-template-parsig in the
    // first place.)
    // The advantage of doing this in HandleTranslationUnit() is that all
    // codegen (when using -add-plugin) is completely finished and this can't
    // affect the compiler output.
    struct Visitor : public RecursiveASTVisitor<Visitor> {
      const std::set<std::string> &ParsedTemplates;
      Visitor(const std::set<std::string> &ParsedTemplates)
          : ParsedTemplates(ParsedTemplates) {}
      bool VisitFunctionDecl(FunctionDecl *FD) {
        if (FD->isLateTemplateParsed() &&
            ParsedTemplates.count(FD->getNameAsString()))
          LateParsedDecls.insert(FD);
        return true;
      }

      std::set<FunctionDecl*> LateParsedDecls;
    } v(ParsedTemplates);
    v.TraverseDecl(context.getTranslationUnitDecl());
    clang::Sema &sema = Instance.getSema();
    for (const FunctionDecl *FD : v.LateParsedDecls) {
      clang::LateParsedTemplate &LPT =
          *sema.LateParsedTemplateMap.find(FD)->second;
      sema.LateTemplateParser(sema.OpaqueParser, LPT);
      llvm::errs() << "late-parsed-decl: \"" << FD->getNameAsString() << "\"\n";
    }
  }
};

class PrintFunctionNamesAction : public PluginASTAction {
  std::set<std::string> ParsedTemplates;
  bool WarnOnDecls = false;
  bool RemarkOnDecls = false;
  bool ErrorOnDecls = false;

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    return std::make_unique<PrintFunctionsConsumer>(
        CI, ParsedTemplates, WarnOnDecls, RemarkOnDecls, ErrorOnDecls);
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      llvm::errs() << "PrintFunctionNames arg = " << args[i] << "\n";

      // Example error handling.
      DiagnosticsEngine &D = CI.getDiagnostics();
      if (args[i] == "-warn-decls") {
        WarnOnDecls = true;
      } else if (args[i] == "-remark-decls") {
        RemarkOnDecls = true;
      } else if (args[i] == "-error-decls") {
        ErrorOnDecls = true;
      } else if (args[i] == "-an-error") {
        unsigned DiagID = D.getCustomDiagID(DiagnosticsEngine::Error,
                                            "invalid argument '%0'");
        D.Report(DiagID) << args[i];
        return false;
      } else if (args[i] == "-parse-template") {
        if (i + 1 >= e) {
          D.Report(D.getCustomDiagID(DiagnosticsEngine::Error,
                                     "missing -parse-template argument"));
          return false;
        }
        ++i;
        ParsedTemplates.insert(args[i]);
      }
    }
    if (!args.empty() && args[0] == "help")
      PrintHelp(llvm::errs());

    return true;
  }
  void PrintHelp(llvm::raw_ostream& ros) {
    ros << "Help for PrintFunctionNames plugin goes here\n";
  }

};

}

static FrontendPluginRegistry::Add<PrintFunctionNamesAction>
X("print-fns", "print function names");
