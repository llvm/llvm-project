//===-- PrototypeTestGen.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/LibcTableGenUtil/APIIndexer.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

namespace {

llvm::cl::opt<std::string> EntrypointsFilename(
    "entrypoints_file",
    llvm::cl::desc("file containing the comma separated list of entrypoints"),
    llvm::cl::Required);

} // anonymous namespace

bool TestGeneratorMain(llvm::raw_ostream &OS, llvm::RecordKeeper &records) {
  OS << "#include \"src/__support/CPP/type_traits.h\"\n";
  llvm_libc::APIIndexer G(records);
  std::unordered_set<std::string> headerFileSet;

  auto Entrypoints = llvm::MemoryBuffer::getFile(EntrypointsFilename);

  std::vector<std::string> entrypoints;
  for (auto entrypoint : llvm::split((*Entrypoints)->getBuffer(), ','))
    entrypoints.push_back(entrypoint.str());
  for (auto entrypoint : entrypoints) {
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
  for (const auto &entrypoint : entrypoints) {
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
  OS << "  // Check that all entrypoints are present in the binary";
  OS << "  uintptr_t check = 0;\n";
  for (const auto &entrypoint : entrypoints) {
    if (entrypoint == "errno")
      continue;
    OS << "  check += reinterpret_cast<uintptr_t>(&" << entrypoint << ");\n";
  }

  OS << '\n';
  OS << "  return check != 0 ? 0 : 1;\n";
  OS << "}\n\n";

  return false;
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], TestGeneratorMain);
}
