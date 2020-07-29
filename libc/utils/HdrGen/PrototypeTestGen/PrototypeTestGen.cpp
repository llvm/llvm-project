//===-- PrototypeTestGen.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/LibcTableGenUtil/APIIndexer.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

namespace {

llvm::cl::list<std::string>
    EntrypointNamesOption("e", llvm::cl::desc("<list of entrypoints>"),
                          llvm::cl::OneOrMore);

} // anonymous namespace

bool TestGeneratorMain(llvm::raw_ostream &OS, llvm::RecordKeeper &records) {
  OS << "#include \"TypeTraits.h\"\n";
  llvm_libc::APIIndexer G(records);
  for (const auto &header : G.PublicHeaders)
    OS << "#include <" << header << ">\n";
  OS << '\n';
  OS << "int main() {\n";
  for (const auto &entrypoint : EntrypointNamesOption) {
    auto match = G.FunctionSpecMap.find(entrypoint);
    if (match == G.FunctionSpecMap.end()) {
      llvm::errs() << "ERROR: entrypoint '" << entrypoint
                   << "' could not be found in spec in any public header\n";
      return true;
    }
    llvm::Record *functionSpec = match->second;
    llvm::Record *retValSpec = functionSpec->getValueAsDef("Return");
    std::string returnType =
        G.getTypeAsString(retValSpec->getValueAsDef("ReturnType"));
    // _Noreturn is an indication for the compiler that a function
    // doesn't return, and isn't a type understood by c++ templates.
    if (llvm::StringRef(returnType).contains("_Noreturn"))
      returnType = "void";

    OS << "  static_assert(__llvm_libc::cpp::IsSame<" << returnType << '(';
    auto args = functionSpec->getValueAsListOfDefs("Args");
    for (size_t i = 0, size = args.size(); i < size; ++i) {
      llvm::Record *argType = args[i]->getValueAsDef("ArgType");
      OS << G.getTypeAsString(argType);
      if (i < size - 1)
        OS << ", ";
    }
    OS << "), decltype(" << entrypoint << ")>::Value, ";
    OS << '"' << entrypoint
       << " prototype in TableGen does not match public header" << '"';
    OS << ");\n";
  }

  OS << '\n';
  OS << "  return 0;\n";
  OS << "}\n\n";

  return false;
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], TestGeneratorMain);
}
