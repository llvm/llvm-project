//===-- tools/extra/clang-reorder-fields/tool/ClangReorderFields.cpp -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of clang-reorder-fields tool
///
//===----------------------------------------------------------------------===//

#include "../ReorderFieldsAction.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include <cstdlib>
#include <string>
#include <system_error>

using namespace llvm;
using namespace clang;

cl::OptionCategory ClangReorderFieldsCategory("clang-reorder-fields options");

static cl::opt<std::string>
    RecordName("record-name", cl::Required,
               cl::desc("The name of the struct/class."),
               cl::cat(ClangReorderFieldsCategory));

static cl::list<std::string> FieldsOrder("fields-order", cl::CommaSeparated,
                                         cl::OneOrMore,
                                         cl::desc("The desired fields order."),
                                         cl::cat(ClangReorderFieldsCategory));

static cl::opt<bool> Inplace("i", cl::desc("Overwrite edited files."),
                             cl::cat(ClangReorderFieldsCategory));

const char Usage[] = "A tool to reorder fields in C/C++ structs/classes.\n";

int main(int argc, const char **argv) {
  auto ExpectedParser = tooling::CommonOptionsParser::create(
      argc, argv, ClangReorderFieldsCategory, cl::OneOrMore, Usage);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }

  tooling::CommonOptionsParser &OP = ExpectedParser.get();

  auto Files = OP.getSourcePathList();
  tooling::RefactoringTool Tool(OP.getCompilations(), Files);

  reorder_fields::ReorderFieldsAction Action(RecordName, FieldsOrder,
                                             Tool.getReplacements());

  auto Factory = tooling::newFrontendActionFactory(&Action);

  if (Inplace)
    return Tool.runAndSave(Factory.get());

  int ExitCode = Tool.run(Factory.get());
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
  TextDiagnosticPrinter DiagnosticPrinter(errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);

  auto &FileMgr = Tool.getFiles();
  SourceManager Sources(Diagnostics, FileMgr);
  Rewriter Rewrite(Sources, DefaultLangOptions);
  Tool.applyAllReplacements(Rewrite);

  for (const auto &File : Files) {
    auto Entry = FileMgr.getFile(File);
    const auto ID = Sources.getOrCreateFileID(*Entry, SrcMgr::C_User);
    Rewrite.getEditBuffer(ID).write(outs());
  }

  return ExitCode;
}
