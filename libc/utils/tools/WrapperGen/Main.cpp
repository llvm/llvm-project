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
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"

#include <fstream>
#include <map>
#include <sstream>
#include <string>

llvm::cl::opt<bool>
    GenWrapper("gen-wrapper",
               llvm::cl::desc("Generate a C wrapper for <name>."));
llvm::cl::opt<bool> GenAlias("gen-alias",
                             llvm::cl::desc("Generate a C alias for <name>."));

llvm::cl::opt<std::string>
    FunctionName("name", llvm::cl::desc("Name of the function to be wrapped."),
                 llvm::cl::value_desc("<function name>"), llvm::cl::Required);
llvm::cl::opt<std::string> MangledNameString(
    "mangled-name", llvm::cl::desc("Declare as an alias to this mangled name."),
    llvm::cl::value_desc("<aliasee string>"));
llvm::cl::opt<std::string> MangledNameFile(
    "mangled-name-file",
    llvm::cl::desc("Declare as an alias to the C name read from "
                   "this file."),
    llvm::cl::value_desc("<path to a file containing alias name>"));
llvm::cl::opt<std::string>
    AppendToFile("append-to-file",
                 llvm::cl::desc("Append the generated content at the end of "
                                "the contents of this file."),
                 llvm::cl::value_desc("<path to a file>"));

void validateOpts() {
  int ActionCount = 0;
  if (GenWrapper)
    ++ActionCount;
  if (GenAlias)
    ++ActionCount;
  if (ActionCount != 1) {
    llvm::PrintFatalError("Exactly one of {--gen-wrapper, --gen-alias} "
                          "should be specified");
  }
  if (!MangledNameString.empty() && !MangledNameFile.empty()) {
    llvm::PrintFatalError("The options 'mangled-name' and 'mangled-name-file' "
                          "cannot be specified simultaneously.");
  }
}

static std::string getMangledName() {
  if (!MangledNameString.empty())
    return MangledNameString;

  if (MangledNameFile.empty())
    llvm::PrintFatalError("At least one of --mangled-name or "
                          "--mangled-name-file should be specified.");

  auto ErrorOrBuf = llvm::MemoryBuffer::getFile(MangledNameFile);
  if (!ErrorOrBuf)
    llvm::PrintFatalError("Unable to read the mangled name file " +
                          MangledNameFile);
  llvm::StringRef FileContent = ErrorOrBuf.get()->getBuffer().trim();
  llvm::SmallVector<llvm::StringRef> Lines;
  FileContent.split(Lines, '\n');
  for (llvm::StringRef L : Lines) {
    if (L.contains("__llvm_libc"))
      return std::string(L);
  }
  llvm::PrintFatalError("Did not find an LLVM libc mangled name in " +
                        MangledNameFile);
  return std::string();
}

void writeAppendToFile(llvm::raw_ostream &OS) {
  auto ErrorOrBuf = llvm::MemoryBuffer::getFile(AppendToFile);
  if (!ErrorOrBuf) {
    llvm::PrintFatalError("Unable to read the file '" + AppendToFile +
                          "' to append to.");
  }
  OS << ErrorOrBuf.get()->getBuffer().trim() << '\n';
}

llvm::Record *getFunctionSpec(const llvm_libc::APIIndexer &Indexer) {
  auto Iter = Indexer.FunctionSpecMap.find(FunctionName);
  if (Iter == Indexer.FunctionSpecMap.end()) {
    llvm::PrintFatalError("Function '" + FunctionName +
                          "' not found in any standard spec.");
  }
  auto &NameSpecPair = *Iter;
  return NameSpecPair.second;
}

std::pair<std::string, bool> writeFunctionHeader(llvm_libc::APIIndexer &Indexer,
                                                 llvm::Record *FunctionSpec,
                                                 llvm::raw_ostream &OS) {
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
  return make_pair(CallArgs.str(), ShouldReturn);
}

static bool generateWrapper(llvm::raw_ostream &OS,
                            llvm::RecordKeeper &Records) {
  llvm_libc::APIIndexer Indexer(Records);
  llvm::Record *FunctionSpec = getFunctionSpec(Indexer);
  if (AppendToFile.empty()) {
    std::string Header = Indexer.FunctionToHeaderMap[FunctionName];
    auto RelPath =
        llvm::StringRef(Header).drop_back(2); // Drop the ".h" suffix.
    OS << "#include \"src/" << RelPath << "/" << FunctionName << ".h\"\n";
  } else {
    writeAppendToFile(OS);
  }
  auto Pair = writeFunctionHeader(Indexer, FunctionSpec, OS);
  OS << ") {\n"
     << "  " << (Pair.second ? "return " : "")
     << "__llvm_libc::" << FunctionName << "(" << Pair.first << ");\n"
     << "}\n";
  return false;
}

static bool generateAlias(llvm::raw_ostream &OS, llvm::RecordKeeper &Records) {
  if (!AppendToFile.empty())
    writeAppendToFile(OS);
  llvm_libc::APIIndexer Indexer(Records);
  llvm::Record *FunctionSpec = getFunctionSpec(Indexer);
  auto Pair = writeFunctionHeader(Indexer, FunctionSpec, OS);
  OS << ") __attribute__((alias(\"" << getMangledName() << "\")));\n";
  return false;
}

static bool wrapperGenMain(llvm::raw_ostream &OS, llvm::RecordKeeper &Records) {
  validateOpts();

  if (GenWrapper)
    return generateWrapper(OS, Records);
  if (GenAlias)
    return generateAlias(OS, Records);

  __builtin_unreachable();
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], wrapperGenMain);
}
