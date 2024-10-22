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

bool TestGeneratorMain(llvm::raw_ostream &OS,
                       const llvm::RecordKeeper &records) {
  OS << "#include \"src/__support/CPP/type_traits.h\"\n";
  llvm_libc::APIIndexer G(records);
  std::unordered_set<std::string> headerFileSet;
  for (const auto &entrypoint : EntrypointNamesOption) {
    if (entrypoint == "errno")
      continue;
    auto match = G.FunctionToHeaderMap.find(entrypoint);
    if (match == G.FunctionToHeaderMap.end()) {
      auto objectMatch = G.ObjectToHeaderMap.find(entrypoint);
      if (objectMatch != G.ObjectToHeaderMap.end()) {
        headerFileSet.insert(objectMatch->second);
        continue;
      }

      llvm::errs() << "ERROR: entrypoint '" << entrypoint
                   << "' could not be found in spec in any public header\n";
      return true;
    }
    headerFileSet.insert(match->second);
  }
  for (const auto &header : headerFileSet)
    OS << "#include <" << header << ">\n";

  OS << '\n';

  OS << "extern \"C\" int main() {\n";
  for (const auto &entrypoint : EntrypointNamesOption) {
    if (entrypoint == "errno")
      continue;
    auto match = G.FunctionSpecMap.find(entrypoint);
    if (match == G.FunctionSpecMap.end()) {
      auto objectMatch = G.ObjectSpecMap.find(entrypoint);
      if (objectMatch != G.ObjectSpecMap.end()) {
        auto entrypointPtr = entrypoint + "_ptr";
        llvm::Record *objectSpec = G.ObjectSpecMap[entrypoint];
        auto objectType = objectSpec->getValueAsString("Type");
        // We just make sure that the global object is present.
        OS << "  " << objectType << " *" << entrypointPtr << " = &"
           << entrypoint << ";\n";
        OS << "  ++" << entrypointPtr << ";\n"; // To avoid unused var warning.
        continue;
      }
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

    OS << "  static_assert(LIBC_NAMESPACE::cpp::is_same_v<" << returnType
       << '(';
    auto args = functionSpec->getValueAsListOfDefs("Args");
    for (size_t i = 0, size = args.size(); i < size; ++i) {
      llvm::Record *argType = args[i]->getValueAsDef("ArgType");
      OS << G.getTypeAsString(argType);
      if (i < size - 1)
        OS << ", ";
    }
    OS << ") __NOEXCEPT, decltype(" << entrypoint << ")>, ";
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
