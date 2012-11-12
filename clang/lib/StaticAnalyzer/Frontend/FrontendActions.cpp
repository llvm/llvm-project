//===--- FrontendActions.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "AnalysisConsumer.h"
using namespace clang;
using namespace ento;

ASTConsumer *AnalysisAction::CreateASTConsumer(CompilerInstance &CI,
                                               StringRef InFile) {
  return CreateAnalysisConsumer(CI.getPreprocessor(),
                                CI.getFrontendOpts().OutputFile,
                                CI.getAnalyzerOpts(),
                                CI.getFrontendOpts().Plugins);
}

