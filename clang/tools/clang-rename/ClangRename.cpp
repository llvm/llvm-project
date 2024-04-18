//===--- tools/extra/clang-rename/ClangRename.cpp - Clang rename tool -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a clang-rename tool that automatically finds and
/// renames symbols in C++ code.
///
//===----------------------------------------------------------------------===//

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
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"
#include "clang/Tooling/Refactoring/Rename/USRFindingAction.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <system_error>

#include "Opts.inc"

using namespace llvm;
using namespace clang;

/// An oldname -> newname rename.
struct RenameAllInfo {
  unsigned Offset = 0;
  std::string QualifiedName;
  std::string NewName;
};

LLVM_YAML_IS_SEQUENCE_VECTOR(RenameAllInfo)

namespace llvm {
namespace yaml {

/// Specialized MappingTraits to describe how a RenameAllInfo is
/// (de)serialized.
template <> struct MappingTraits<RenameAllInfo> {
  static void mapping(IO &IO, RenameAllInfo &Info) {
    IO.mapOptional("Offset", Info.Offset);
    IO.mapOptional("QualifiedName", Info.QualifiedName);
    IO.mapRequired("NewName", Info.NewName);
  }
};

} // end namespace yaml
} // end namespace llvm

namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE)                                                    \
  static constexpr llvm::StringLiteral NAME##_init[] = VALUE;                  \
  static constexpr llvm::ArrayRef<llvm::StringLiteral> NAME(                   \
      NAME##_init, std::size(NAME##_init) - 1);
#include "Opts.inc"
#undef PREFIX

using namespace llvm::opt;
static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

class ClangRenameOptTable : public opt::GenericOptTable {
public:
  ClangRenameOptTable() : GenericOptTable(InfoTable) {}
};
} // end anonymous namespace

static cl::OptionCategory ClangRenameOptions("clang-rename common options");

static std::vector<unsigned> SymbolOffsets;
static bool Inplace;
static std::vector<std::string> QualifiedNames;
static std::vector<std::string> NewNames;
static bool PrintName;
static bool PrintLocations;
static std::string ExportFixes;
static std::string Input;
static bool Force;

static tooling::CommonOptionsParser::Args ParseArgs(int argc,
                                                    const char **argv) {
  ClangRenameOptTable Tbl;
  llvm::StringRef ToolName = argv[0];
  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver{A};
  llvm::opt::InputArgList Args = Tbl.parseArgs(
      argc, const_cast<char **>(argv), OPT_UNKNOWN, Saver, [&](StringRef Msg) {
        WithColor::error() << Msg << "\n";
        std::exit(1);
      });

  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(llvm::outs(), "clang-rename [options]", "clang-rename");
    std::exit(0);
  }
  if (Args.hasArg(OPT_version)) {
    llvm::outs() << ToolName << '\n';
    llvm::cl::PrintVersionMessage();
    std::exit(0);
  }

  for (const llvm::opt::Arg *A : Args.filtered(OPT_offset_EQ)) {
    StringRef S{A->getValue()};
    unsigned Value;
    if (!llvm::to_integer(S, Value, 0)) {
      WithColor::error() << ToolName << ": for the --offset option: '" << S
                         << "' value invalid for uint argument!\n";
      std::exit(1);
    }
    SymbolOffsets.emplace_back(Value);
  }

  Inplace = Args.hasArg(OPT_inplace);

  for (const llvm::opt::Arg *A : Args.filtered(OPT_qualified_name_EQ))
    QualifiedNames.emplace_back(A->getValue());
  for (const llvm::opt::Arg *A : Args.filtered(OPT_new_name_EQ))
    NewNames.emplace_back(A->getValue());

  PrintName = Args.hasArg(OPT_print_name);
  PrintLocations = Args.hasArg(OPT_print_locations);

  if (const llvm::opt::Arg *A = Args.getLastArg(OPT_export_fixes_EQ))
    ExportFixes = A->getValue();
  if (const llvm::opt::Arg *A = Args.getLastArg(OPT_input_EQ))
    Input = A->getValue();

  Force = Args.hasArg(OPT_force);

  tooling::CommonOptionsParser::Args args;
  if (const llvm::opt::Arg *A = Args.getLastArg(OPT_build_path_EQ))
    args.BuildPath = A->getValue();
  for (const llvm::opt::Arg *A : Args.filtered(OPT_extra_arg_EQ))
    args.ArgsAfter.emplace_back(A->getValue());
  for (const llvm::opt::Arg *A : Args.filtered(OPT_extra_arg_before_EQ))
    args.ArgsBefore.emplace_back(A->getValue());
  for (const llvm::opt::Arg *A : Args.filtered(OPT_INPUT))
    args.SourcePaths.emplace_back(A->getValue());
  if (args.SourcePaths.empty()) {
    WithColor::error() << ToolName
                       << ": must set at least one source path (-p).\n";
    std::exit(1);
  }
  return args;
}

int main(int argc, const char **argv) {
  auto callback = [&](int &argc, const char **argv)
      -> llvm::Expected<tooling::CommonOptionsParser::Args> {
    return ParseArgs(argc, argv);
  };

  auto ExpectedParser = tooling::CommonOptionsParser::create(
      argc, const_cast<const char **>(argv), callback);
  if (!ExpectedParser) {
    WithColor::error() << ExpectedParser.takeError();
    return 1;
  }
  tooling::CommonOptionsParser &OP = ExpectedParser.get();

  if (!Input.empty()) {
    // Populate QualifiedNames and NewNames from a YAML file.
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
        llvm::MemoryBuffer::getFile(Input);
    if (!Buffer) {
      WithColor::error() << "clang-rename: failed to read " << Input << ": "
                         << Buffer.getError().message() << "\n";
      return 1;
    }

    std::vector<RenameAllInfo> Infos;
    llvm::yaml::Input YAML(Buffer.get()->getBuffer());
    YAML >> Infos;
    for (const auto &Info : Infos) {
      if (!Info.QualifiedName.empty())
        QualifiedNames.push_back(Info.QualifiedName);
      else
        SymbolOffsets.push_back(Info.Offset);
      NewNames.push_back(Info.NewName);
    }
  }

  // Check the arguments for correctness.
  if (NewNames.empty()) {
    WithColor::error() << "clang-rename: -new-name must be specified.\n\n";
    return 1;
  }

  if (SymbolOffsets.empty() == QualifiedNames.empty()) {
    WithColor::error()
        << "clang-rename: -offset and -qualified-name can't be present at "
           "the same time.\n";
    return 1;
  }

  // Check if NewNames is a valid identifier in C++17.
  LangOptions Options;
  Options.CPlusPlus = true;
  Options.CPlusPlus17 = true;
  IdentifierTable Table(Options);
  for (const auto &NewName : NewNames) {
    auto NewNameTokKind = Table.get(NewName).getTokenID();
    if (!tok::isAnyIdentifier(NewNameTokKind)) {
      WithColor::error()
          << "ERROR: new name is not a valid identifier in C++17.\n\n";
      return 1;
    }
  }

  if (SymbolOffsets.size() + QualifiedNames.size() != NewNames.size()) {
    WithColor::error() << "clang-rename: number of symbol offsets("
                       << SymbolOffsets.size()
                       << ") + number of qualified names ("
                       << QualifiedNames.size()
                       << ") must be equal to number of new names("
                       << NewNames.size() << ").\n\n";
    cl::PrintHelpMessage();
    return 1;
  }

  auto Files = OP.getSourcePathList();
  tooling::RefactoringTool Tool(OP.getCompilations(), Files);
  tooling::USRFindingAction FindingAction(SymbolOffsets, QualifiedNames, Force);
  Tool.run(tooling::newFrontendActionFactory(&FindingAction).get());
  const std::vector<std::vector<std::string>> &USRList =
      FindingAction.getUSRList();
  const std::vector<std::string> &PrevNames = FindingAction.getUSRSpellings();
  if (PrintName) {
    for (const auto &PrevName : PrevNames) {
      outs() << "clang-rename found name: " << PrevName << '\n';
    }
  }

  if (FindingAction.errorOccurred()) {
    // Diagnostics are already issued at this point.
    return 1;
  }

  // Perform the renaming.
  tooling::RenamingAction RenameAction(NewNames, PrevNames, USRList,
                                       Tool.getReplacements(), PrintLocations);
  std::unique_ptr<tooling::FrontendActionFactory> Factory =
      tooling::newFrontendActionFactory(&RenameAction);
  int ExitCode;

  if (Inplace) {
    ExitCode = Tool.runAndSave(Factory.get());
  } else {
    ExitCode = Tool.run(Factory.get());

    if (!ExportFixes.empty()) {
      std::error_code EC;
      llvm::raw_fd_ostream OS(ExportFixes, EC, llvm::sys::fs::OF_None);
      if (EC) {
        WithColor::error() << "Error opening output file: " << EC.message()
                           << '\n';
        return 1;
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
      return 0;
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
      auto Entry = FileMgr.getOptionalFileRef(File);
      if (!Entry) {
        WithColor::error() << "clang-rename: " << File << " does not exist.\n";
        return 1;
      }
      const auto ID = Sources.getOrCreateFileID(*Entry, SrcMgr::C_User);
      Rewrite.getEditBuffer(ID).write(outs());
    }
  }

  return ExitCode;
}
