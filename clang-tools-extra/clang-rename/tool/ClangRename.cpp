//===--- tools/extra/clang-rename/ClangRename.cpp - Clang rename tool -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a clang-rename tool that automatically finds and
/// renames symbols in C++ code.
///
//===----------------------------------------------------------------------===//

#include "../RenamingAction.h"
#include "../USRFindingAction.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>
#include <system_error>

using namespace llvm;

using namespace clang;

cl::OptionCategory ClangRenameAtCategory("clang-rename rename-at options");
cl::OptionCategory ClangRenameAllCategory("clang-rename rename-all options");

const char RenameAtUsage[] = "A tool to rename symbols in C/C++ code.\n\
clang-rename renames every occurrence of a symbol found at <offset> in\n\
<source0>. If -i is specified, the edited files are overwritten to disk.\n\
Otherwise, the results are written to stdout.\n";

const char RenameAllUsage[] = "A tool to rename symbols in C/C++ code.\n\
clang-rename performs renaming given pairs {offset | old-name} -> new-name.\n";

static int renameAtMain(int argc, const char *argv[]);
static int renameAllMain(int argc, const char *argv[]);
static int helpMain(int argc, const char *argv[]);

/// \brief An oldname -> newname rename.
struct RenameAllInfo {
  std::string OldName;
  unsigned Offset = 0;
  std::string NewName;
};

LLVM_YAML_IS_SEQUENCE_VECTOR(RenameAllInfo)

namespace llvm {
namespace yaml {

/// \brief Specialized MappingTraits to describe how a RenameAllInfo is
/// (de)serialized.
template <> struct MappingTraits<RenameAllInfo> {
  static void mapping(IO &IO, RenameAllInfo &Info) {
    IO.mapOptional("OldName", Info.OldName);
    IO.mapOptional("Offset", Info.Offset);
    IO.mapRequired("NewName", Info.NewName);
  }
};

} // end namespace yaml
} // end namespace llvm

int main(int argc, const char **argv) {
  if (argc > 1) {
    using MainFunction = std::function<int(int, const char *[])>;
    MainFunction Func = StringSwitch<MainFunction>(argv[1])
                            .Case("rename-at", renameAtMain)
                            .Case("rename-all", renameAllMain)
                            .Cases("-help", "--help", helpMain)
                            .Default(nullptr);

    if (Func) {
      std::string Invocation = std::string(argv[0]) + " " + argv[1];
      argv[1] = Invocation.c_str();
      return Func(argc - 1, argv + 1);
    } else {
      return renameAtMain(argc, argv);
    }
  }

  helpMain(argc, argv);
  return 1;
}

int subcommandMain(bool isRenameAll, int argc, const char **argv) {
  cl::OptionCategory *Category = nullptr;
  const char *Usage = nullptr;
  if (isRenameAll) {
    Category = &ClangRenameAllCategory;
    Usage = RenameAllUsage;
  } else {
    Category = &ClangRenameAtCategory;
    Usage = RenameAtUsage;
  }

  cl::list<std::string> NewNames(
      "new-name", cl::desc("The new name to change the symbol to."),
      (isRenameAll ? cl::ZeroOrMore : cl::Required), cl::cat(*Category));
  cl::list<unsigned> SymbolOffsets(
      "offset",
      cl::desc("Locates the symbol by offset as opposed to <line>:<column>."),
      (isRenameAll ? cl::ZeroOrMore : cl::Required), cl::cat(*Category));
  cl::list<std::string> OldNames(
      "old-name",
      cl::desc(
          "The fully qualified name of the symbol, if -offset is not used."),
      (isRenameAll ? cl::ZeroOrMore : cl::Optional),
      cl::cat(ClangRenameAllCategory));
  cl::opt<bool> Inplace("i", cl::desc("Overwrite edited <file>s."),
                        cl::cat(*Category));
  cl::opt<bool> PrintName(
      "pn",
      cl::desc("Print the found symbol's name prior to renaming to stderr."),
      cl::cat(ClangRenameAtCategory));
  cl::opt<bool> PrintLocations(
      "pl", cl::desc("Print the locations affected by renaming to stderr."),
      cl::cat(ClangRenameAtCategory));
  cl::opt<std::string> ExportFixes(
      "export-fixes", cl::desc("YAML file to store suggested fixes in."),
      cl::value_desc("filename"), cl::cat(*Category));
  cl::opt<std::string> Input(
      "input", cl::desc("YAML file to load oldname-newname pairs from."),
      cl::Optional, cl::cat(ClangRenameAllCategory));

  tooling::CommonOptionsParser OP(argc, argv, *Category, Usage);

  if (!Input.empty()) {
    // Populate OldNames and NewNames from a YAML file.
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
        llvm::MemoryBuffer::getFile(Input);
    if (!Buffer) {
      errs() << "clang-rename: failed to read " << Input << ": "
             << Buffer.getError().message() << "\n";
      return 1;
    }

    std::vector<RenameAllInfo> Infos;
    llvm::yaml::Input YAML(Buffer.get()->getBuffer());
    YAML >> Infos;
    for (const auto &Info : Infos) {
      if (!Info.OldName.empty())
        OldNames.push_back(Info.OldName);
      else
        SymbolOffsets.push_back(Info.Offset);
      NewNames.push_back(Info.NewName);
    }
  }

  // Check the arguments for correctness.

  if (NewNames.empty()) {
    errs() << "clang-rename: either -new-name or -input is required.\n\n";
    exit(1);
  }

  // Check if NewNames is a valid identifier in C++17.
  for (const auto &NewName : NewNames) {
    LangOptions Options;
    Options.CPlusPlus = true;
    Options.CPlusPlus1z = true;
    IdentifierTable Table(Options);
    auto NewNameTokKind = Table.get(NewName).getTokenID();
    if (!tok::isAnyIdentifier(NewNameTokKind)) {
      errs() << "ERROR: new name is not a valid identifier in C++17.\n\n";
      exit(1);
    }
  }

  if (!OldNames.empty() && OldNames.size() != NewNames.size()) {
    errs() << "clang-rename: number of old names (" << OldNames.size()
           << ") do not equal to number of new names (" << NewNames.size()
           << ").\n\n";
    cl::PrintHelpMessage();
    exit(1);
  }

  if (!SymbolOffsets.empty() && SymbolOffsets.size() != NewNames.size()) {
    errs() << "clang-rename: number of symbol offsets (" << SymbolOffsets.size()
           << ") do not equal to number of new names (" << NewNames.size()
           << ").\n\n";
    cl::PrintHelpMessage();
    exit(1);
  }

  std::vector<std::vector<std::string>> USRList;
  std::vector<std::string> PrevNames;
  auto Files = OP.getSourcePathList();
  tooling::RefactoringTool Tool(OP.getCompilations(), Files);
  unsigned Count = OldNames.size() ? OldNames.size() : SymbolOffsets.size();
  for (unsigned I = 0; I < Count; ++I) {
    unsigned SymbolOffset = SymbolOffsets.empty() ? 0 : SymbolOffsets[I];
    const std::string &OldName = OldNames.empty() ? std::string() : OldNames[I];

    // Get the USRs.
    rename::USRFindingAction USRAction(SymbolOffset, OldName);

    // Find the USRs.
    Tool.run(tooling::newFrontendActionFactory(&USRAction).get());
    const auto &USRs = USRAction.getUSRs();
    USRList.push_back(USRs);
    const auto &PrevName = USRAction.getUSRSpelling();
    PrevNames.push_back(PrevName);

    if (PrevName.empty()) {
      // An error should have already been printed.
      exit(1);
    }

    if (PrintName) {
      errs() << "clang-rename: found name: " << PrevName << '\n';
    }
  }

  // Perform the renaming.
  rename::RenamingAction RenameAction(NewNames, PrevNames, USRList,
                                      Tool.getReplacements(), PrintLocations);
  auto Factory = tooling::newFrontendActionFactory(&RenameAction);
  int ExitCode;

  if (Inplace) {
    ExitCode = Tool.runAndSave(Factory.get());
  } else {
    ExitCode = Tool.run(Factory.get());

    if (!ExportFixes.empty()) {
      std::error_code EC;
      llvm::raw_fd_ostream OS(ExportFixes, EC, llvm::sys::fs::F_None);
      if (EC) {
        llvm::errs() << "Error opening output file: " << EC.message() << '\n';
        exit(1);
      }

      // Export replacements.
      tooling::TranslationUnitReplacements TUR;
      const auto &FileToReplacements = Tool.getReplacements();
      for (const auto &Entry : FileToReplacements)
        TUR.Replacements.insert(TUR.Replacements.end(), Entry.second.begin(),
                                Entry.second.end());

      yaml::Output YAML(OS);
      YAML << TUR;
      OS.close();
      exit(0);
    }

    // Write every file to stdout. Right now we just barf the files without any
    // indication of which files start where, other than that we print the files
    // in the same order we see them.
    LangOptions DefaultLangOptions;
    IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
    TextDiagnosticPrinter DiagnosticPrinter(errs(), &*DiagOpts);
    DiagnosticsEngine Diagnostics(
        IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
        &DiagnosticPrinter, false);
    auto &FileMgr = Tool.getFiles();
    SourceManager Sources(Diagnostics, FileMgr);
    Rewriter Rewrite(Sources, DefaultLangOptions);

    Tool.applyAllReplacements(Rewrite);
    for (const auto &File : Files) {
      const auto *Entry = FileMgr.getFile(File);
      auto ID = Sources.translateFile(Entry);
      Rewrite.getEditBuffer(ID).write(outs());
    }
  }

  exit(ExitCode);
}

/// \brief Top level help.
/// FIXME It would be better if this could be auto-generated.
static int helpMain(int argc, const char *argv[]) {
  errs() << "Usage: clang-rename {rename-at|rename-all} [OPTION]...\n\n"
            "A tool to rename symbols in C/C++ code.\n\n"
            "Subcommands:\n"
            "  rename-at:  Perform rename off of a location in a file. (This "
            "is the default.)\n"
            "  rename-all: Perform rename of all symbols matching one or more "
            "fully qualified names.\n";
  return 0;
}

static int renameAtMain(int argc, const char *argv[]) {
  return subcommandMain(false, argc, argv);
}

static int renameAllMain(int argc, const char *argv[]) {
  return subcommandMain(true, argc, argv);
}
