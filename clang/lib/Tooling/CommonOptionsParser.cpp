//===--- CommonOptionsParser.cpp - common options for clang tools ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the CommonOptionsParser class used to parse common
//  command-line options for clang tools, so that they can be run as separate
//  command-line applications with a consistent common interface for handling
//  compilation database and input files.
//
//  It provides a common subset of command-line options, common algorithm
//  for locating a compilation database and source files, and help messages
//  for the basic command-line interface.
//
//  It creates a CompilationDatabase and reads common command-line options.
//
//  This class uses the Clang Tooling infrastructure, see
//    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
//  for details on setting it up with LLVM source tree.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

using namespace clang::tooling;
using namespace llvm;

const char *const CommonOptionsParser::HelpMessage =
    "\n"
    "-p <build-path> is used to read a compile command database.\n"
    "\n"
    "\tFor example, it can be a CMake build directory in which a file named\n"
    "\tcompile_commands.json exists (use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\n"
    "\tCMake option to get this output). When no build path is specified,\n"
    "\ta search for compile_commands.json will be attempted through all\n"
    "\tparent paths of the first input file . See:\n"
    "\thttp://clang.llvm.org/docs/HowToSetupToolingForLLVM.html for an\n"
    "\texample of setting up Clang Tooling on a source tree.\n"
    "\n"
    "<source0> ... specify the paths of source files. These paths are\n"
    "\tlooked up in the compile command database. If the path of a file is\n"
    "\tabsolute, it needs to point into CMake's source tree. If the path is\n"
    "\trelative, the current working directory needs to be in the CMake\n"
    "\tsource tree and the file must be in a subdirectory of the current\n"
    "\tworking directory. \"./\" prefixes in the relative files will be\n"
    "\tautomatically removed, but the rest of a relative path must be a\n"
    "\tsuffix of a path in the compile command database.\n"
    "\n";

namespace {
class ArgumentsAdjustingCompilations : public CompilationDatabase {
public:
  ArgumentsAdjustingCompilations(
      std::unique_ptr<CompilationDatabase> Compilations)
      : Compilations(std::move(Compilations)) {}

  void appendArgumentsAdjuster(ArgumentsAdjuster Adjuster) {
    Adjusters.push_back(std::move(Adjuster));
  }

  std::vector<CompileCommand>
  getCompileCommands(StringRef FilePath) const override {
    return adjustCommands(Compilations->getCompileCommands(FilePath));
  }

  std::vector<std::string> getAllFiles() const override {
    return Compilations->getAllFiles();
  }

  std::vector<CompileCommand> getAllCompileCommands() const override {
    return adjustCommands(Compilations->getAllCompileCommands());
  }

private:
  std::unique_ptr<CompilationDatabase> Compilations;
  std::vector<ArgumentsAdjuster> Adjusters;

  std::vector<CompileCommand>
  adjustCommands(std::vector<CompileCommand> Commands) const {
    for (CompileCommand &Command : Commands)
      for (const auto &Adjuster : Adjusters)
        Command.CommandLine = Adjuster(Command.CommandLine, Command.Filename);
    return Commands;
  }
};
} // namespace

CommonOptionsParser::CommonOptionsParser(
    int &argc, const char **argv, cl::OptionCategory &Category,
    llvm::cl::NumOccurrencesFlag OccurrencesFlag, const char *Overview) {
  static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

  static cl::opt<std::string> BuildPath("p", cl::desc("Build path"),
                                        cl::Optional, cl::cat(Category));

  static cl::list<std::string> SourcePaths(
      cl::Positional, cl::desc("<source0> [... <sourceN>]"), OccurrencesFlag,
      cl::cat(Category));

  static cl::list<std::string> ArgsAfter(
      "extra-arg",
      cl::desc("Additional argument to append to the compiler command line"),
      cl::cat(Category));

  static cl::list<std::string> ArgsBefore(
      "extra-arg-before",
      cl::desc("Additional argument to prepend to the compiler command line"),
      cl::cat(Category));

  cl::HideUnrelatedOptions(Category);

  Compilations.reset(FixedCompilationDatabase::loadFromCommandLine(argc, argv));
  cl::ParseCommandLineOptions(argc, argv, Overview);
  cl::PrintOptionValues();

  SourcePathList = SourcePaths;
  if ((OccurrencesFlag == cl::ZeroOrMore || OccurrencesFlag == cl::Optional) &&
      SourcePathList.empty())
    return;
  if (!Compilations) {
    std::string ErrorMessage;
    if (!BuildPath.empty()) {
      Compilations =
          CompilationDatabase::autoDetectFromDirectory(BuildPath, ErrorMessage);
    } else {
      Compilations = CompilationDatabase::autoDetectFromSource(SourcePaths[0],
                                                               ErrorMessage);
    }
    if (!Compilations) {
      llvm::errs() << "Error while trying to load a compilation database:\n"
                   << ErrorMessage << "Running without flags.\n";
      Compilations.reset(
          new FixedCompilationDatabase(".", std::vector<std::string>()));
    }
  }
  auto AdjustingCompilations =
      llvm::make_unique<ArgumentsAdjustingCompilations>(
          std::move(Compilations));
  AdjustingCompilations->appendArgumentsAdjuster(
      getInsertArgumentAdjuster(ArgsBefore, ArgumentInsertPosition::BEGIN));
  AdjustingCompilations->appendArgumentsAdjuster(
      getInsertArgumentAdjuster(ArgsAfter, ArgumentInsertPosition::END));
  Compilations = std::move(AdjustingCompilations);
}
