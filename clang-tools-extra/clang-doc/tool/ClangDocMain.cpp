//===-- ClangDocMain.cpp - ClangDoc -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool for generating C and C++ documentation from source code
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
#include "clang/Tooling/AllTUsExecution.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <mutex>
#include <string>

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

static llvm::cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static llvm::cl::OptionCategory ClangDocCategory("clang-doc options");

static llvm::cl::opt<std::string>
    ProjectName("project-name", llvm::cl::desc("Name of project."),
                llvm::cl::cat(ClangDocCategory));

static llvm::cl::opt<bool> IgnoreMappingFailures(
    "ignore-map-errors",
    llvm::cl::desc("Continue if files are not mapped correctly."),
    llvm::cl::init(true), llvm::cl::cat(ClangDocCategory));

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

static llvm::cl::opt<std::string> UserAssetPath(
    "asset",
    llvm::cl::desc("User supplied asset path to "
                   "override the default css and js files for html output"),
    llvm::cl::cat(ClangDocCategory));

static llvm::cl::opt<std::string> SourceRoot("source-root", llvm::cl::desc(R"(
Directory where processed files are stored.
Links to definition locations will only be
generated if the file is in this dir.)"),
                                             llvm::cl::cat(ClangDocCategory));

static llvm::cl::opt<std::string>
    RepositoryUrl("repository", llvm::cl::desc(R"(
URL of repository that hosts code.
Used for links to definition locations.)"),
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
std::string getExecutablePath(const char *Argv0, void *MainAddr) {
  return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
}

llvm::Error getAssetFiles(clang::doc::ClangDocContext &CDCtx) {
  using DirIt = llvm::sys::fs::directory_iterator;
  std::error_code FileErr;
  llvm::SmallString<128> FilePath(UserAssetPath);
  for (DirIt DirStart = DirIt(UserAssetPath, FileErr),
                   DirEnd;
       !FileErr && DirStart != DirEnd; DirStart.increment(FileErr)) {
    FilePath = DirStart->path();
    if (llvm::sys::fs::is_regular_file(FilePath)) {
      if (llvm::sys::path::extension(FilePath) == ".css")
        CDCtx.UserStylesheets.insert(CDCtx.UserStylesheets.begin(),
                                     std::string(FilePath));
      else if (llvm::sys::path::extension(FilePath) == ".js")
        CDCtx.JsScripts.emplace_back(FilePath.str());
    }
  }
  if (FileErr)
    return llvm::createFileError(FilePath, FileErr);
  return llvm::Error::success();
}

llvm::Error getDefaultAssetFiles(const char *Argv0,
                                 clang::doc::ClangDocContext &CDCtx) {
  void *MainAddr = (void *)(intptr_t)getExecutablePath;
  std::string ClangDocPath = getExecutablePath(Argv0, MainAddr);
  llvm::SmallString<128> NativeClangDocPath;
  llvm::sys::path::native(ClangDocPath, NativeClangDocPath);

  llvm::SmallString<128> AssetsPath;
  AssetsPath = llvm::sys::path::parent_path(NativeClangDocPath);
  llvm::sys::path::append(AssetsPath, "..", "share", "clang-doc");
  llvm::SmallString<128> DefaultStylesheet;
  llvm::sys::path::native(AssetsPath, DefaultStylesheet);
  llvm::sys::path::append(DefaultStylesheet,
                          "clang-doc-default-stylesheet.css");
  llvm::SmallString<128> IndexJS;
  llvm::sys::path::native(AssetsPath, IndexJS);
  llvm::sys::path::append(IndexJS, "index.js");

  if (!llvm::sys::fs::is_regular_file(IndexJS))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "default index.js file missing at " +
                                       IndexJS + "\n");

  if (!llvm::sys::fs::is_regular_file(DefaultStylesheet))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "default clang-doc-default-stylesheet.css file missing at " +
            DefaultStylesheet + "\n");

  CDCtx.UserStylesheets.insert(CDCtx.UserStylesheets.begin(),
                               std::string(DefaultStylesheet));
  CDCtx.JsScripts.emplace_back(IndexJS.str());

  return llvm::Error::success();
}

llvm::Error getHtmlAssetFiles(const char *Argv0,
                              clang::doc::ClangDocContext &CDCtx) {
  if (!UserAssetPath.empty() &&
      !llvm::sys::fs::is_directory(std::string(UserAssetPath)))
    llvm::outs() << "Asset path supply is not a directory: " << UserAssetPath
                 << " falling back to default\n";
  if (llvm::sys::fs::is_directory(std::string(UserAssetPath)))
    return getAssetFiles(CDCtx);
  return getDefaultAssetFiles(Argv0, CDCtx);
}

/// Make the output of clang-doc deterministic by sorting the children of
/// namespaces and records.
void sortUsrToInfo(llvm::StringMap<std::unique_ptr<doc::Info>> &USRToInfo) {
  for (auto &I : USRToInfo) {
    auto &Info = I.second;
    if (Info->IT == doc::InfoType::IT_namespace) {
      auto *Namespace = static_cast<clang::doc::NamespaceInfo *>(Info.get());
      Namespace->Children.sort();
    }
    if (Info->IT == doc::InfoType::IT_record) {
      auto *Record = static_cast<clang::doc::RecordInfo *>(Info.get());
      Record->Children.sort();
    }
  }
}

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  std::error_code OK;

  const char *Overview =
      R"(Generates documentation from source code and comments.

Example usage for files without flags (default):

  $ clang-doc File1.cpp File2.cpp ... FileN.cpp

Example usage for a project using a compile commands database:

  $ clang-doc --executor=all-TUs compile_commands.json
)";

  auto Executor = clang::tooling::createExecutorFromCommandLineArgs(
      argc, argv, ClangDocCategory, Overview);

  if (!Executor) {
    llvm::errs() << toString(Executor.takeError()) << "\n";
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
      Executor->get()->getExecutionContext(),
      ProjectName,
      PublicOnly,
      OutDirectory,
      SourceRoot,
      RepositoryUrl,
      {UserStylesheets.begin(), UserStylesheets.end()}
  };

  if (Format == "html") {
    if (auto Err = getHtmlAssetFiles(argv[0], CDCtx)) {
      llvm::errs() << toString(std::move(Err)) << "\n";
      return 1;
    }
  }

  // Mapping phase
  llvm::outs() << "Mapping decls...\n";
  auto Err =
      Executor->get()->execute(doc::newMapperActionFactory(CDCtx), ArgAdjuster);
  if (Err) {
    if (IgnoreMappingFailures)
      llvm::errs() << "Error mapping decls in files. Clang-doc will ignore "
                      "these files and continue:\n"
                   << toString(std::move(Err)) << "\n";
    else {
      llvm::errs() << toString(std::move(Err)) << "\n";
      return 1;
    }
  }

  // Collect values into output by key.
  // In ToolResults, the Key is the hashed USR and the value is the
  // bitcode-encoded representation of the Info object.
  llvm::outs() << "Collecting infos...\n";
  llvm::StringMap<std::vector<StringRef>> USRToBitcode;
  Executor->get()->getToolResults()->forEachResult(
      [&](StringRef Key, StringRef Value) {
        auto R = USRToBitcode.try_emplace(Key, std::vector<StringRef>());
        R.first->second.emplace_back(Value);
      });

  // Collects all Infos according to their unique USR value. This map is added
  // to from the thread pool below and is protected by the USRToInfoMutex.
  llvm::sys::Mutex USRToInfoMutex;
  llvm::StringMap<std::unique_ptr<doc::Info>> USRToInfo;

  // First reducing phase (reduce all decls into one info per decl).
  llvm::outs() << "Reducing " << USRToBitcode.size() << " infos...\n";
  std::atomic<bool> Error;
  Error = false;
  llvm::sys::Mutex IndexMutex;
  // ExecutorConcurrency is a flag exposed by AllTUsExecution.h
  llvm::DefaultThreadPool Pool(llvm::hardware_concurrency(ExecutorConcurrency));
  for (auto &Group : USRToBitcode) {
    Pool.async([&]() {
      std::vector<std::unique_ptr<doc::Info>> Infos;
      for (auto &Bitcode : Group.getValue()) {
        llvm::BitstreamCursor Stream(Bitcode);
        doc::ClangDocBitcodeReader Reader(Stream);
        auto ReadInfos = Reader.readBitcode();
        if (!ReadInfos) {
          llvm::errs() << toString(ReadInfos.takeError()) << "\n";
          Error = true;
          return;
        }
        std::move(ReadInfos->begin(), ReadInfos->end(),
                  std::back_inserter(Infos));
      }

      auto Reduced = doc::mergeInfos(Infos);
      if (!Reduced) {
        llvm::errs() << llvm::toString(Reduced.takeError());
        return;
      }

      // Add a reference to this Info in the Index
      {
        std::lock_guard<llvm::sys::Mutex> Guard(IndexMutex);
        clang::doc::Generator::addInfoToIndex(CDCtx.Idx, Reduced.get().get());
      }

      // Save in the result map (needs a lock due to threaded access).
      {
        std::lock_guard<llvm::sys::Mutex> Guard(USRToInfoMutex);
        USRToInfo[Group.getKey()] = std::move(Reduced.get());
      }
    });
  }

  Pool.wait();

  if (Error)
    return 1;

  sortUsrToInfo(USRToInfo);

  // Ensure the root output directory exists.
  if (std::error_code Err = llvm::sys::fs::create_directories(OutDirectory);
      Err != std::error_code()) {
    llvm::errs() << "Failed to create directory '" << OutDirectory << "'\n";
    return 1;
  }

  // Run the generator.
  llvm::outs() << "Generating docs...\n";
  if (auto Err =
          G->get()->generateDocs(OutDirectory, std::move(USRToInfo), CDCtx)) {
    llvm::errs() << toString(std::move(Err)) << "\n";
    return 1;
  }

  llvm::outs() << "Generating assets for docs...\n";
  Err = G->get()->createResources(CDCtx);
  if (Err) {
    llvm::outs() << "warning: " << toString(std::move(Err)) << "\n";
  }

  return 0;
}
