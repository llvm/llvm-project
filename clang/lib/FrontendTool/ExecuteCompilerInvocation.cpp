//===--- ExecuteCompilerInvocation.cpp ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file holds ExecuteCompilerInvocation(). It is split into its own file to
// minimize the impact of pulling in essentially everything else in Clang.
//
//===----------------------------------------------------------------------===//

#include "clang/FrontendTool/Utils.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "clang/ARCMigrate/ARCMTActions.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/OptTable.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Rewrite/Frontend/FrontendActions.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/DynamicLibrary.h"
using namespace clang;

static FrontendAction *CreateFrontendBaseAction(CompilerInstance &CI) {
  using namespace clang::frontend;

  switch (CI.getFrontendOpts().ProgramAction) {
  case ASTDeclList:            return new ASTDeclListAction();
  case ASTDump:                return new ASTDumpAction();
  case ASTDumpXML:             return new ASTDumpXMLAction();
  case ASTPrint:               return new ASTPrintAction();
  case ASTView:                return new ASTViewAction();
  case DumpRawTokens:          return new DumpRawTokensAction();
  case DumpTokens:             return new DumpTokensAction();
  case EmitAssembly:           return new EmitAssemblyAction();
  case EmitBC:                 return new EmitBCAction();
  case EmitHTML:               return new HTMLPrintAction();
  case EmitLLVM:               return new EmitLLVMAction();
  case EmitLLVMOnly:           return new EmitLLVMOnlyAction();
  case EmitCodeGenOnly:        return new EmitCodeGenOnlyAction();
  case EmitObj:                return new EmitObjAction();
  case FixIt:                  return new FixItAction();
  case GenerateModule:         return new GenerateModuleAction;
  case GeneratePCH:            return new GeneratePCHAction;
  case GeneratePTH:            return new GeneratePTHAction();
  case InitOnly:               return new InitOnlyAction();
  case ParseSyntaxOnly:        return new SyntaxOnlyAction();

  case PluginAction: {
    for (FrontendPluginRegistry::iterator it =
           FrontendPluginRegistry::begin(), ie = FrontendPluginRegistry::end();
         it != ie; ++it) {
      if (it->getName() == CI.getFrontendOpts().ActionName) {
        OwningPtr<PluginASTAction> P(it->instantiate());
        if (!P->ParseArgs(CI, CI.getFrontendOpts().PluginArgs))
          return 0;
        return P.take();
      }
    }

    CI.getDiagnostics().Report(diag::err_fe_invalid_plugin_name)
      << CI.getFrontendOpts().ActionName;
    return 0;
  }

  case PrintDeclContext:       return new DeclContextPrintAction();
  case PrintPreamble:          return new PrintPreambleAction();
  case PrintPreprocessedInput: {
    if (CI.getPreprocessorOutputOpts().RewriteIncludes)
      return new RewriteIncludesAction();
    return new PrintPreprocessedAction();
  }

  case RewriteMacros:          return new RewriteMacrosAction();
  case RewriteObjC:            return new RewriteObjCAction();
  case RewriteTest:            return new RewriteTestAction();
  case RunAnalysis:            return new ento::AnalysisAction();
  case MigrateSource:          return new arcmt::MigrateSourceAction();
  case RunPreprocessorOnly:    return new PreprocessOnlyAction();
  }
  llvm_unreachable("Invalid program action!");
}

static FrontendAction *CreateFrontendAction(CompilerInstance &CI) {
  // Create the underlying action.
  FrontendAction *Act = CreateFrontendBaseAction(CI);
  if (!Act)
    return 0;

  const FrontendOptions &FEOpts = CI.getFrontendOpts();

  if (FEOpts.FixAndRecompile) {
    Act = new FixItRecompile(Act);
  }
  
  // Potentially wrap the base FE action in an ARC Migrate Tool action.
  switch (FEOpts.ARCMTAction) {
  case FrontendOptions::ARCMT_None:
    break;
  case FrontendOptions::ARCMT_Check:
    Act = new arcmt::CheckAction(Act);
    break;
  case FrontendOptions::ARCMT_Modify:
    Act = new arcmt::ModifyAction(Act);
    break;
  case FrontendOptions::ARCMT_Migrate:
    Act = new arcmt::MigrateAction(Act,
                                   FEOpts.MTMigrateDir,
                                   FEOpts.ARCMTMigrateReportOut,
                                   FEOpts.ARCMTMigrateEmitARCErrors);
    break;
  }

  if (FEOpts.ObjCMTAction != FrontendOptions::ObjCMT_None) {
    Act = new arcmt::ObjCMigrateAction(Act, FEOpts.MTMigrateDir,
                   FEOpts.ObjCMTAction & ~FrontendOptions::ObjCMT_Literals,
                   FEOpts.ObjCMTAction & ~FrontendOptions::ObjCMT_Subscripting);
  }

  // If there are any AST files to merge, create a frontend action
  // adaptor to perform the merge.
  if (!FEOpts.ASTMergeFiles.empty())
    Act = new ASTMergeAction(Act, FEOpts.ASTMergeFiles);

  return Act;
}

bool clang::ExecuteCompilerInvocation(CompilerInstance *Clang) {
  // Honor -help.
  if (Clang->getFrontendOpts().ShowHelp) {
    OwningPtr<driver::OptTable> Opts(driver::createDriverOptTable());
    Opts->PrintHelp(llvm::outs(), "clang -cc1",
                    "LLVM 'Clang' Compiler: http://clang.llvm.org",
                    /*Include=*/driver::options::CC1Option,
                    /*Exclude=*/0);
    return 0;
  }

  // Honor -version.
  //
  // FIXME: Use a better -version message?
  if (Clang->getFrontendOpts().ShowVersion) {
    llvm::cl::PrintVersionMessage();
    return 0;
  }

  // Load any requested plugins.
  for (unsigned i = 0,
         e = Clang->getFrontendOpts().Plugins.size(); i != e; ++i) {
    const std::string &Path = Clang->getFrontendOpts().Plugins[i];
    std::string Error;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(Path.c_str(), &Error))
      Clang->getDiagnostics().Report(diag::err_fe_unable_to_load_plugin)
        << Path << Error;
  }

  // Honor -mllvm.
  //
  // FIXME: Remove this, one day.
  // This should happen AFTER plugins have been loaded!
  if (!Clang->getFrontendOpts().LLVMArgs.empty()) {
    unsigned NumArgs = Clang->getFrontendOpts().LLVMArgs.size();
    const char **Args = new const char*[NumArgs + 2];
    Args[0] = "clang (LLVM option parsing)";
    for (unsigned i = 0; i != NumArgs; ++i)
      Args[i + 1] = Clang->getFrontendOpts().LLVMArgs[i].c_str();
    Args[NumArgs + 1] = 0;
    llvm::cl::ParseCommandLineOptions(NumArgs + 1, Args);
  }

  // Honor -analyzer-checker-help.
  // This should happen AFTER plugins have been loaded!
  if (Clang->getAnalyzerOpts()->ShowCheckerHelp) {
    ento::printCheckerHelp(llvm::outs(), Clang->getFrontendOpts().Plugins);
    return 0;
  }

  // If there were errors in processing arguments, don't do anything else.
  bool Success = false;
  if (!Clang->getDiagnostics().hasErrorOccurred()) {
    // Create and execute the frontend action.
    OwningPtr<FrontendAction> Act(CreateFrontendAction(*Clang));
    if (Act) {
      Success = Clang->ExecuteAction(*Act);
      if (Clang->getFrontendOpts().DisableFree)
        Act.take();
    }
  }

  return Success;
}
