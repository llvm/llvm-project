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
    "\tclang-check will attempt to locate it automatically using all parent\n"
    "\tpaths of the first input file. See:\n"
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

CommonOptionsParser::CommonOptionsParser(int &argc, const char **argv) {
  static cl::opt<std::string> BuildPath(
      "p", cl::desc("Build path"), cl::Optional);

  static cl::list<std::string> SourcePaths(
      cl::Positional, cl::desc("<source0> [... <sourceN>]"), cl::OneOrMore);

  Compilations.reset(FixedCompilationDatabase::loadFromCommandLine(argc,
                                                                   argv));
  cl::ParseCommandLineOptions(argc, argv);
  SourcePathList = SourcePaths;
  if (!Compilations) {
    std::string ErrorMessage;
    if (!BuildPath.empty()) {
      Compilations.reset(CompilationDatabase::autoDetectFromDirectory(
                              BuildPath, ErrorMessage));
    } else {
      Compilations.reset(CompilationDatabase::autoDetectFromSource(
                              SourcePaths[0], ErrorMessage));
    }
    if (!Compilations)
      llvm::report_fatal_error(ErrorMessage);
  }
}
