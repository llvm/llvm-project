//===-- ClangMoveMain.cpp - move defintion to new file ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangMove.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/YAMLTraits.h"
#include <set>
#include <string>

using namespace clang;
using namespace llvm;

namespace {

std::error_code CreateNewFile(const llvm::Twine &path) {
  int fd = 0;
  if (std::error_code ec =
          llvm::sys::fs::openFileForWrite(path, fd, llvm::sys::fs::F_Text))
    return ec;

  return llvm::sys::Process::SafelyCloseFileDescriptor(fd);
}

cl::OptionCategory ClangMoveCategory("clang-move options");

cl::list<std::string> Names("names", cl::CommaSeparated, cl::OneOrMore,
                            cl::desc("The list of the names of classes being "
                                     "moved, e.g. \"Foo,a::Foo,b::Foo\"."),
                            cl::cat(ClangMoveCategory));

cl::opt<std::string>
    OldHeader("old_header",
              cl::desc("The relative/absolute file path of old header."),
              cl::cat(ClangMoveCategory));

cl::opt<std::string>
    OldCC("old_cc", cl::desc("The relative/absolute file path of old cc."),
          cl::cat(ClangMoveCategory));

cl::opt<std::string>
    NewHeader("new_header",
              cl::desc("The relative/absolute file path of new header."),
              cl::cat(ClangMoveCategory));

cl::opt<std::string>
    NewCC("new_cc", cl::desc("The relative/absolute file path of new cc."),
          cl::cat(ClangMoveCategory));

cl::opt<bool> OldDependOnNew(
    "old_depend_on_new",
    cl::desc(
        "Whether old header will depend on new header. If true, clang-move will "
        "add #include of new header to old header."),
    cl::init(false), cl::cat(ClangMoveCategory));

cl::opt<bool> NewDependOnOld(
    "new_depend_on_old",
    cl::desc(
        "Whether new header will depend on old header. If true, clang-move will "
        "add #include of old header to new header."),
    cl::init(false), cl::cat(ClangMoveCategory));

cl::opt<std::string>
    Style("style",
          cl::desc("The style name used for reformatting. Default is \"llvm\""),
          cl::init("llvm"), cl::cat(ClangMoveCategory));

cl::opt<bool> Dump("dump_result",
                   cl::desc("Dump results in JSON format to stdout."),
                   cl::cat(ClangMoveCategory));

} // namespace

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  // Add "-fparse-all-comments" compile option to make clang parse all comments,
  // otherwise, ordinary comments like "//" and "/*" won't get parsed (This is
  // a bit of hacky).
  std::vector<std::string> ExtraArgs(argv, argv + argc);
  ExtraArgs.insert(ExtraArgs.begin() + 1, "-extra-arg=-fparse-all-comments");
  std::unique_ptr<const char *[]> RawExtraArgs(
      new const char *[ExtraArgs.size()]);
  for (size_t i = 0; i < ExtraArgs.size(); ++i)
    RawExtraArgs[i] = ExtraArgs[i].c_str();
  int Argc = argc + 1;
  tooling::CommonOptionsParser OptionsParser(Argc, RawExtraArgs.get(),
                                             ClangMoveCategory);

  if (OldDependOnNew && NewDependOnOld) {
    llvm::errs() << "Provide either --old_depend_on_new or "
                    "--new_depend_on_old. clang-move doesn't support these two "
                    "options at same time (It will introduce include cycle).\n";
    return 1;
  }

  tooling::RefactoringTool Tool(OptionsParser.getCompilations(),
                                OptionsParser.getSourcePathList());
  move::ClangMoveTool::MoveDefinitionSpec Spec;
  Spec.Names = {Names.begin(), Names.end()};
  Spec.OldHeader = OldHeader;
  Spec.NewHeader = NewHeader;
  Spec.OldCC = OldCC;
  Spec.NewCC = NewCC;
  Spec.OldDependOnNew = OldDependOnNew;
  Spec.NewDependOnOld = NewDependOnOld;

  llvm::SmallString<128> InitialDirectory;
  if (std::error_code EC = llvm::sys::fs::current_path(InitialDirectory))
    llvm::report_fatal_error("Cannot detect current path: " +
                             Twine(EC.message()));

  auto Factory = llvm::make_unique<clang::move::ClangMoveActionFactory>(
      Spec, Tool.getReplacements(), InitialDirectory.str(), Style);

  int CodeStatus = Tool.run(Factory.get());
  if (CodeStatus)
    return CodeStatus;

  if (!NewCC.empty()) {
    std::error_code EC = CreateNewFile(NewCC);
    if (EC) {
      llvm::errs() << "Failed to create " << NewCC << ": " << EC.message()
                   << "\n";
      return EC.value();
    }
  }
  if (!NewHeader.empty()) {
    std::error_code EC = CreateNewFile(NewHeader);
    if (EC) {
      llvm::errs() << "Failed to create " << NewHeader << ": " << EC.message()
                   << "\n";
      return EC.value();
    }
  }

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
  clang::TextDiagnosticPrinter DiagnosticPrinter(errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);
  auto &FileMgr = Tool.getFiles();
  SourceManager SM(Diagnostics, FileMgr);
  Rewriter Rewrite(SM, LangOptions());

  if (!formatAndApplyAllReplacements(Tool.getReplacements(), Rewrite, Style)) {
    llvm::errs() << "Failed applying all replacements.\n";
    return 1;
  }

  if (Dump) {
    std::set<llvm::StringRef> Files;
    for (const auto &it : Tool.getReplacements())
      Files.insert(it.first);
    auto WriteToJson = [&](llvm::raw_ostream &OS) {
      OS << "[\n";
      for (auto File : Files) {
        OS << "  {\n";
        OS << "    \"FilePath\": \"" << File << "\",\n";
        const auto *Entry = FileMgr.getFile(File);
        auto ID = SM.translateFile(Entry);
        std::string Content;
        llvm::raw_string_ostream ContentStream(Content);
        Rewrite.getEditBuffer(ID).write(ContentStream);
        OS << "    \"SourceText\": \""
           << llvm::yaml::escape(ContentStream.str()) << "\"\n";
        OS << "  }";
        if (File != *(--Files.end()))
          OS << ",\n";
      }
      OS << "\n]\n";
    };
    WriteToJson(llvm::outs());
    return 0;
  }

  return Rewrite.overwriteChangedFiles();
}
