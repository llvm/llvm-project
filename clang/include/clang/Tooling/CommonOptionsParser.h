//===- CommonOptionsParser.h - common options for clang tools -*- C++ -*-=====//
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

#ifndef LLVM_TOOLS_CLANG_INCLUDE_CLANG_TOOLING_COMMONOPTIONSPARSER_H
#define LLVM_TOOLS_CLANG_INCLUDE_CLANG_TOOLING_COMMONOPTIONSPARSER_H

#include "clang/Tooling/CompilationDatabase.h"

namespace clang {
namespace tooling {
/// \brief A parser for options common to all command-line Clang tools.
///
/// Parses a common subset of command-line arguments, locates and loads a
/// compilation commands database and runs a tool with user-specified action. It
/// also contains a help message for the common command-line options.
///
/// An example of usage:
/// \code
/// #include "clang/Frontend/FrontendActions.h"
/// #include "clang/Tooling/CommonOptionsParser.h"
/// #include "llvm/Support/CommandLine.h"
///
/// using namespace clang::tooling;
/// using namespace llvm;
///
/// static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
/// static cl::extrahelp MoreHelp("\nMore help text...");
/// static cl:opt<bool> YourOwnOption(...);
/// ...
///
/// int main(int argc, const char **argv) {
///   CommonOptionsParser OptionsParser(argc, argv);
///   ClangTool Tool(OptionsParser.GetCompilations(),
///                  OptionsParser.GetSourcePathListi());
///   return Tool.run(newFrontendActionFactory<clang::SyntaxOnlyAction>());
/// }
/// \endcode
class CommonOptionsParser {
public:
  /// \brief Parses command-line, initializes a compilation database.
  /// This constructor can change argc and argv contents, e.g. consume
  /// command-line options used for creating FixedCompilationDatabase.
  /// This constructor exits program in case of error.
  CommonOptionsParser(int &argc, const char **argv);

  /// Returns a reference to the loaded compilations database.
  CompilationDatabase &GetCompilations() {
    return *Compilations;
  }

  /// Returns a list of source file paths to process.
  std::vector<std::string> GetSourcePathList() {
    return SourcePathList;
  }

  static const char *const HelpMessage;

private:
  llvm::OwningPtr<CompilationDatabase> Compilations;
  std::vector<std::string> SourcePathList;
};

}  // namespace tooling
}  // namespace clang

#endif  // LLVM_TOOLS_CLANG_INCLUDE_CLANG_TOOLING_COMMONOPTIONSPARSER_H
