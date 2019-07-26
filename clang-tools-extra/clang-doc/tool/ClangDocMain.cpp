//===-- ClangDocMain.cpp - ClangDoc -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool for generating C and C++ documenation from source code
// and comments. Generally, it runs a LibTooling FrontendAction on source files,
// mapping each declaration in those files to its USR and serializing relevant
// information into LLVM bitcode. It then runs a pass over the collected
// declaration information, reducing by USR. There is an option to dump this
// intermediate result to bitcode. Finally, it hands the reduced information
// off to a generator, which does the final parsing from the intermediate
// representation to the desired output format.
//
//===----------------------------------------------------------------------===//

#include "BitcodeReader.h"
#include "BitcodeWriter.h"
#include "ClangDoc.h"
#include "Generators.h"
#include "Representation.h"
#include "clang/AST/AST.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

static llvm::cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static llvm::cl::OptionCategory ClangDocCategory("clang-doc options");

static llvm::cl::opt<std::string>
    OutDirectory("output",
                 llvm::cl::desc("Directory for outputting generated files."),
                 llvm::cl::init("docs"), llvm::cl::cat(ClangDocCategory));

static llvm::cl::opt<bool>
    PublicOnly("public", llvm::cl::desc("Document only public declarations."),
               llvm::cl::init(false), llvm::cl::cat(ClangDocCategory));

static llvm::cl::opt<bool> DoxygenOnly(
    "doxygen",
    llvm::cl::desc("Use only doxygen-style comments to generate docs."),
    llvm::cl::init(false), llvm::cl::cat(ClangDocCategory));

static llvm::cl::list<std::string> UserStylesheets(
    "stylesheets", llvm::cl::CommaSeparated,
    llvm::cl::desc("CSS stylesheets to extend the default styles."),
    llvm::cl::cat(ClangDocCategory));

enum OutputFormatTy {
  md,
  yaml,
  html,
};

static llvm::cl::opt<OutputFormatTy>
    FormatEnum("format", llvm::cl::desc("Format for outputted docs."),
               llvm::cl::values(clEnumValN(OutputFormatTy::yaml, "yaml",
                                           "Documentation in YAML format."),
                                clEnumValN(OutputFormatTy::md, "md",
                                           "Documentation in MD format."),
                                clEnumValN(OutputFormatTy::html, "html",
                                           "Documentation in HTML format.")),
               llvm::cl::init(OutputFormatTy::yaml),
               llvm::cl::cat(ClangDocCategory));

std::string getFormatString() {
  switch (FormatEnum) {
  case OutputFormatTy::yaml:
    return "yaml";
  case OutputFormatTy::md:
    return "md";
  case OutputFormatTy::html:
    return "html";
  }
  llvm_unreachable("Unknown OutputFormatTy");
}

// This function isn't referenced outside its translation unit, but it
// can't use the "static" keyword because its address is used for
// GetMainExecutable (since some platforms don't support taking the
// address of main, and some platforms can't implement GetMainExecutable
// without being given the address of a function in the main executable).
std::string GetExecutablePath(const char *Argv0, void *MainAddr) {
  return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
}

bool CreateDirectory(const Twine &DirName, bool ClearDirectory = false) {
  std::error_code OK;
  llvm::SmallString<128> DocsRootPath;
  if (ClearDirectory) {
    std::error_code RemoveStatus = llvm::sys::fs::remove_directories(DirName);
    if (RemoveStatus != OK) {
      llvm::errs() << "Unable to remove existing documentation directory for "
                   << DirName << ".\n";
      return true;
    }
  }
  std::error_code DirectoryStatus = llvm::sys::fs::create_directories(DirName);
  if (DirectoryStatus != OK) {
    llvm::errs() << "Unable to create documentation directories.\n";
    return true;
  }
  return false;
}

// A function to extract the appropriate file name for a given info's
// documentation. The path returned is a composite of the output directory, the
// info's relative path and name and the extension. The relative path should
// have been constructed in the serialization phase.
//
// Example: Given the below, the <ext> path for class C will be
// <root>/A/B/C.<ext>
//
// namespace A {
// namesapce B {
//
// class C {};
//
// }
// }
llvm::Expected<llvm::SmallString<128>> getInfoOutputFile(StringRef Root,
                                                         StringRef RelativePath,
                                                         StringRef Name,
                                                         StringRef Ext) {
  llvm::SmallString<128> Path;
  llvm::sys::path::native(Root, Path);
  llvm::sys::path::append(Path, RelativePath);
  if (CreateDirectory(Path))
    return llvm::make_error<llvm::StringError>("Unable to create directory.\n",
                                               llvm::inconvertibleErrorCode());
  llvm::sys::path::append(Path, Name + Ext);
  return Path;
}

// Iterate through tool results and build string map of info vectors from the
// encoded bitstreams.
bool bitcodeResultsToInfos(
    tooling::ToolResults &Results,
    llvm::StringMap<std::vector<std::unique_ptr<doc::Info>>> &Output) {
  bool Err = false;
  Results.forEachResult([&](StringRef Key, StringRef Value) {
    llvm::BitstreamCursor Stream(Value);
    doc::ClangDocBitcodeReader Reader(Stream);
    auto Infos = Reader.readBitcode();
    if (!Infos) {
      llvm::errs() << toString(Infos.takeError()) << "\n";
      Err = true;
      return;
    }
    for (auto &I : Infos.get()) {
      auto R =
          Output.try_emplace(Key, std::vector<std::unique_ptr<doc::Info>>());
      R.first->second.emplace_back(std::move(I));
    }
  });
  return Err;
}

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  std::error_code OK;

  ExecutorName.setInitialValue("all-TUs");
  auto Exec = clang::tooling::createExecutorFromCommandLineArgs(
      argc, argv, ClangDocCategory);

  if (!Exec) {
    llvm::errs() << toString(Exec.takeError()) << "\n";
    return 1;
  }

  // Fail early if an invalid format was provided.
  std::string Format = getFormatString();
  llvm::outs() << "Emiting docs in " << Format << " format.\n";
  auto G = doc::findGeneratorByName(Format);
  if (!G) {
    llvm::errs() << toString(G.takeError()) << "\n";
    return 1;
  }

  ArgumentsAdjuster ArgAdjuster;
  if (!DoxygenOnly)
    ArgAdjuster = combineAdjusters(
        getInsertArgumentAdjuster("-fparse-all-comments",
                                  tooling::ArgumentInsertPosition::END),
        ArgAdjuster);

  clang::doc::ClangDocContext CDCtx = {
      Exec->get()->getExecutionContext(),
      PublicOnly,
      OutDirectory,
      {UserStylesheets.begin(), UserStylesheets.end()}};

  if (Format == "html") {
    void *MainAddr = (void *)(intptr_t)GetExecutablePath;
    std::string ClangDocPath = GetExecutablePath(argv[0], MainAddr);
    llvm::SmallString<128> DefaultStylesheet;
    llvm::sys::path::native(ClangDocPath, DefaultStylesheet);
    DefaultStylesheet = llvm::sys::path::parent_path(DefaultStylesheet);
    llvm::sys::path::append(DefaultStylesheet,
                            "../share/clang/clang-doc-default-stylesheet.css");
    CDCtx.UserStylesheets.insert(CDCtx.UserStylesheets.begin(),
                                 DefaultStylesheet.str());
  }

  // Mapping phase
  llvm::outs() << "Mapping decls...\n";
  auto Err =
      Exec->get()->execute(doc::newMapperActionFactory(CDCtx), ArgAdjuster);
  if (Err) {
    llvm::errs() << toString(std::move(Err)) << "\n";
    return 1;
  }

  // Collect values into output by key.
  // In ToolResults, the Key is the hashed USR and the value is the
  // bitcode-encoded representation of the Info object.
  llvm::outs() << "Collecting infos...\n";
  llvm::StringMap<std::vector<std::unique_ptr<doc::Info>>> USRToInfos;
  if (bitcodeResultsToInfos(*Exec->get()->getToolResults(), USRToInfos))
    return 1;

  // First reducing phase (reduce all decls into one info per decl).
  llvm::outs() << "Reducing " << USRToInfos.size() << " infos...\n";
  for (auto &Group : USRToInfos) {
    auto Reduced = doc::mergeInfos(Group.getValue());
    if (!Reduced) {
      llvm::errs() << llvm::toString(Reduced.takeError());
      continue;
    }

    doc::Info *I = Reduced.get().get();
    auto InfoPath = getInfoOutputFile(OutDirectory, I->Path, I->extractName(),
                                      "." + Format);
    if (!InfoPath) {
      llvm::errs() << toString(InfoPath.takeError()) << "\n";
      return 1;
    }
    std::error_code FileErr;
    llvm::raw_fd_ostream InfoOS(InfoPath.get(), FileErr, llvm::sys::fs::F_None);
    if (FileErr != OK) {
      llvm::errs() << "Error opening info file: " << FileErr.message() << "\n";
      continue;
    }

    if (auto Err = G->get()->generateDocForInfo(I, InfoOS, CDCtx))
      llvm::errs() << toString(std::move(Err)) << "\n";
  }

  if (!G->get()->createResources(CDCtx))
    return 1;

  return 0;
}
