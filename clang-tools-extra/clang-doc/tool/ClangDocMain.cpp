//===-- ClangDocMain.cpp - ClangDoc -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

enum OutputFormatTy {
  md,
  yaml,
};

static llvm::cl::opt<OutputFormatTy>
    FormatEnum("format", llvm::cl::desc("Format for outputted docs."),
               llvm::cl::values(clEnumValN(OutputFormatTy::yaml, "yaml",
                                           "Documentation in YAML format."),
                                clEnumValN(OutputFormatTy::md, "md",
                                           "Documentation in MD format.")),
               llvm::cl::init(OutputFormatTy::yaml),
               llvm::cl::cat(ClangDocCategory));

std::string getFormatString() {
  switch (FormatEnum) {
  case OutputFormatTy::yaml:
    return "yaml";
  case OutputFormatTy::md:
    return "md";
  }
  llvm_unreachable("Unknown OutputFormatTy");
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

// A function to extract the appropriate path name for a given info's
// documentation. The path returned is a composite of the parent namespaces as
// directories plus the decl name as the filename.
//
// Example: Given the below, the <ext> path for class C will be <
// root>/A/B/C.<ext>
//
// namespace A {
// namesapce B {
//
// class C {};
//
// }
// }
llvm::Expected<llvm::SmallString<128>>
getInfoOutputFile(StringRef Root,
                  llvm::SmallVectorImpl<doc::Reference> &Namespaces,
                  StringRef Name, StringRef Ext) {
  std::error_code OK;
  llvm::SmallString<128> Path;
  llvm::sys::path::native(Root, Path);
  for (auto R = Namespaces.rbegin(), E = Namespaces.rend(); R != E; ++R)
    llvm::sys::path::append(Path, R->Name);

  if (CreateDirectory(Path))
    return llvm::make_error<llvm::StringError>("Unable to create directory.\n",
                                               llvm::inconvertibleErrorCode());

  if (Name.empty())
    Name = "GlobalNamespace";
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

  // Mapping phase
  llvm::outs() << "Mapping decls...\n";
  clang::doc::ClangDocContext CDCtx = {Exec->get()->getExecutionContext(),
                                       PublicOnly};
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

    auto InfoPath =
        getInfoOutputFile(OutDirectory, I->Namespace, I->Name, "." + Format);
    if (!InfoPath) {
      llvm::errs() << toString(InfoPath.takeError()) << "\n";
      continue;
    }
    std::error_code FileErr;
    llvm::raw_fd_ostream InfoOS(InfoPath.get(), FileErr, llvm::sys::fs::F_None);
    if (FileErr != OK) {
      llvm::errs() << "Error opening info file: " << FileErr.message() << "\n";
      continue;
    }

    if (auto Err = G->get()->generateDocForInfo(I, InfoOS))
      llvm::errs() << toString(std::move(Err)) << "\n";
  }

  return 0;
}
