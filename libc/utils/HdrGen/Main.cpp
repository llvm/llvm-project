//===-- "main" function of libc-hdrgen ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Generator.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Main.h"

#include <string>
#include <unordered_map>

static llvm::cl::opt<std::string>
    HeaderDefFile("def", llvm::cl::desc("Path to the .h.def file."),
                  llvm::cl::value_desc("<filename>"), llvm::cl::Required);
static llvm::cl::opt<std::string> StandardHeader(
    "header",
    llvm::cl::desc("The standard header file which is to be generated."),
    llvm::cl::value_desc("<header file>"));
static llvm::cl::list<std::string> EntrypointNamesOption(
    "e", llvm::cl::value_desc("<list of entrypoints>"),
    llvm::cl::desc(
        "Each --e is one entrypoint (generated from entrypoints.txt)"),
    llvm::cl::OneOrMore);
static llvm::cl::list<std::string> ReplacementValues(
    "args", llvm::cl::desc("Command separated <argument name>=<value> pairs."),
    llvm::cl::value_desc("<name=value>[,name=value]"));
static llvm::cl::opt<bool> ExportDecls(
    "export-decls",
    llvm::cl::desc("Output a new header containing only the entrypoints."));

static void
ParseArgValuePairs(std::unordered_map<std::string, std::string> &Map) {
  for (std::string &R : ReplacementValues) {
    auto Pair = llvm::StringRef(R).split('=');
    Map[std::string(Pair.first)] = std::string(Pair.second);
  }
}

static bool HeaderGeneratorMain(llvm::raw_ostream &OS,
                                const llvm::RecordKeeper &Records) {
  std::unordered_map<std::string, std::string> ArgMap;
  ParseArgValuePairs(ArgMap);
  llvm_libc::Generator G(HeaderDefFile, EntrypointNamesOption, StandardHeader,
                         ArgMap);
  if (ExportDecls)
    G.generateDecls(OS, Records);
  else
    G.generate(OS, Records);

  return false;
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &HeaderGeneratorMain);
}
