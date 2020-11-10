//===-- "main" function of libc-wrappergen --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/LibcTableGenUtil/APIIndexer.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"

#include <sstream>
#include <string>

llvm::cl::opt<std::string>
    FunctionName("name", llvm::cl::desc("Name of the function to be wrapped."),
                 llvm::cl::value_desc("<function name>"), llvm::cl::Required);

static bool WrapperGenMain(llvm::raw_ostream &OS, llvm::RecordKeeper &Records) {
  llvm_libc::APIIndexer Indexer(Records);
  auto Iter = Indexer.FunctionSpecMap.find(FunctionName);
  if (Iter == Indexer.FunctionSpecMap.end()) {
    llvm::PrintFatalError("Function '" + FunctionName +
                          "' not found in any standard spec.");
  }

  // To avoid all confusion, we include the implementation header using the
  // full path (relative the libc directory.)
  std::string Header = Indexer.FunctionToHeaderMap[FunctionName];
  auto RelPath = llvm::StringRef(Header).drop_back(2); // Drop the ".h" suffix.
  OS << "#include \"src/" << RelPath << "/" << FunctionName << ".h\"\n";

  auto &NameSpecPair = *Iter;
  llvm::Record *FunctionSpec = NameSpecPair.second;
  llvm::Record *RetValSpec = FunctionSpec->getValueAsDef("Return");
  llvm::Record *ReturnType = RetValSpec->getValueAsDef("ReturnType");
  std::string ReturnTypeString = Indexer.getTypeAsString(ReturnType);
  bool ShouldReturn = true;
  // We are generating C wrappers in C++ code. So, we should convert the C
  // _Noreturn to the C++ [[noreturn]].
  llvm::StringRef NR("_Noreturn "); // Note the space after _Noreturn
  llvm::StringRef RT(ReturnTypeString);
  if (RT.startswith(NR)) {
    RT = RT.drop_front(NR.size() - 1); // - 1 because of the space.
    ReturnTypeString = std::string("[[noreturn]]") + std::string(RT);
    ShouldReturn = false;
  }
  OS << "extern \"C\" " << ReturnTypeString << " " << FunctionName << "(";

  auto ArgsList = FunctionSpec->getValueAsListOfDefs("Args");
  std::stringstream CallArgs;
  std::string ArgPrefix("__arg");
  for (size_t i = 0; i < ArgsList.size(); ++i) {
    llvm::Record *ArgType = ArgsList[i]->getValueAsDef("ArgType");
    auto TypeName = Indexer.getTypeAsString(ArgType);

    if (TypeName.compare("void") == 0) {
      if (ArgsList.size() == 1) {
        break;
      } else {
        // the reason this is a fatal error is that a void argument means this
        // function has no arguments; multiple copies of no arguments is an
        // error.
        llvm::PrintFatalError(
            "The specification for function " + FunctionName +
            " lists other arguments along with a void argument.");
      }
    }

    OS << TypeName << " " << ArgPrefix << i;
    CallArgs << ArgPrefix << i;
    if (i < ArgsList.size() - 1) {
      OS << ", ";
      CallArgs << ", ";
    }
  }

  // TODO: Arg types of the C++ implementation functions need not
  // match the standard types. Either handle such differences here, or
  // avoid such a thing in the implementations.
  OS << ") {\n"
     << "  " << (ShouldReturn ? "return " : "")
     << "__llvm_libc::" << FunctionName << "(" << CallArgs.str() << ");\n"
     << "}\n";

  return false;
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], WrapperGenMain);
}
