//===---------------- "main" function of libc-hdrgen ------------*- C++ -*-===//
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

namespace {

llvm::cl::opt<std::string>
    HeaderDefFile("def", llvm::cl::desc("Path to the .h.def file."),
                  llvm::cl::value_desc("<filename>"), llvm::cl::Required);
llvm::cl::opt<std::string> StandardHeader(
    "header",
    llvm::cl::desc("The standard header file which is to be generated."),
    llvm::cl::value_desc("<header file>"));
llvm::cl::list<std::string> ReplacementValues(
    "args", llvm::cl::desc("Command seperated <argument name>=<value> pairs."),
    llvm::cl::value_desc("<name=value>[,name=value]"));

void ParseArgValuePairs(std::unordered_map<std::string, std::string> &Map) {
  for (std::string &R : ReplacementValues) {
    auto Pair = llvm::StringRef(R).split('=');
    Map[Pair.first] = Pair.second;
  }
}

} // anonymous namespace

namespace llvm_libc {

bool HeaderGeneratorMain(llvm::raw_ostream &OS, llvm::RecordKeeper &Records) {
  std::unordered_map<std::string, std::string> ArgMap;
  ParseArgValuePairs(ArgMap);
  Generator G(HeaderDefFile, StandardHeader, ArgMap);
  G.generate(OS, Records);

  return false;
}

} // namespace llvm_libc

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &llvm_libc::HeaderGeneratorMain);
}
