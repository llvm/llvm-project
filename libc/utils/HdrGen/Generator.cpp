//===-- Implementation of the main header generation class ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Generator.h"

#include "IncludeFileCommand.h"
#include "PublicAPICommand.h"
#include "utils/LibcTableGenUtil/APIIndexer.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>

static const char CommandPrefix[] = "%%";
static const size_t CommandPrefixSize = llvm::StringRef(CommandPrefix).size();

static const char CommentPrefix[] = "<!>";

static const char ParamNamePrefix[] = "${";
static const size_t ParamNamePrefixSize =
    llvm::StringRef(ParamNamePrefix).size();
static const char ParamNameSuffix[] = "}";
static const size_t ParamNameSuffixSize =
    llvm::StringRef(ParamNameSuffix).size();

namespace llvm_libc {

Command *Generator::getCommandHandler(llvm::StringRef CommandName) {
  if (CommandName == IncludeFileCommand::Name) {
    if (!IncludeFileCmd)
      IncludeFileCmd = std::make_unique<IncludeFileCommand>();
    return IncludeFileCmd.get();
  } else if (CommandName == PublicAPICommand::Name) {
    if (!PublicAPICmd)
      PublicAPICmd = std::make_unique<PublicAPICommand>(EntrypointNameList);
    return PublicAPICmd.get();
  } else {
    return nullptr;
  }
}

void Generator::parseCommandArgs(llvm::StringRef ArgStr, ArgVector &Args) {
  if (!ArgStr.contains(',') && ArgStr.trim(' ').trim('\t').size() == 0) {
    // If it is just space between the parenthesis
    return;
  }

  ArgStr.split(Args, ",");
  for (llvm::StringRef &A : Args) {
    A = A.trim(' ');
    if (A.starts_with(ParamNamePrefix) && A.ends_with(ParamNameSuffix)) {
      A = A.drop_front(ParamNamePrefixSize).drop_back(ParamNameSuffixSize);
      A = ArgMap[std::string(A)];
    }
  }
}

void Generator::generate(llvm::raw_ostream &OS,
                         const llvm::RecordKeeper &Records) {
  auto DefFileBuffer = llvm::MemoryBuffer::getFile(HeaderDefFile);
  if (!DefFileBuffer) {
    llvm::errs() << "Unable to open " << HeaderDefFile << ".\n";
    std::exit(1);
  }
  llvm::SourceMgr SrcMgr;
  unsigned DefFileID = SrcMgr.AddNewSourceBuffer(
      std::move(DefFileBuffer.get()), llvm::SMLoc::getFromPointer(nullptr));

  llvm::StringRef Content = SrcMgr.getMemoryBuffer(DefFileID)->getBuffer();
  while (true) {
    std::pair<llvm::StringRef, llvm::StringRef> P = Content.split('\n');
    Content = P.second;

    llvm::StringRef Line = P.first.trim(' ');
    if (Line.starts_with(CommandPrefix)) {
      Line = Line.drop_front(CommandPrefixSize);

      P = Line.split("(");
      // It's possible that we have windows line endings, so strip off the extra
      // CR.
      P.second = P.second.trim();
      if (P.second.empty() || P.second[P.second.size() - 1] != ')') {
        SrcMgr.PrintMessage(llvm::SMLoc::getFromPointer(P.second.data()),
                            llvm::SourceMgr::DK_Error,
                            "Command argument list should begin with '(' "
                            "and end with ')'.");
        SrcMgr.PrintMessage(llvm::SMLoc::getFromPointer(P.second.data()),
                            llvm::SourceMgr::DK_Error, P.second.data());
        SrcMgr.PrintMessage(llvm::SMLoc::getFromPointer(P.second.data()),
                            llvm::SourceMgr::DK_Error,
                            std::to_string(P.second.size()));
        std::exit(1);
      }
      llvm::StringRef CommandName = P.first;
      Command *Cmd = getCommandHandler(CommandName);
      if (Cmd == nullptr) {
        SrcMgr.PrintMessage(llvm::SMLoc::getFromPointer(CommandName.data()),
                            llvm::SourceMgr::DK_Error,
                            "Unknown command '%%" + CommandName + "'.");
        std::exit(1);
      }

      llvm::StringRef ArgStr = P.second.drop_back(1);
      ArgVector Args;
      parseCommandArgs(ArgStr, Args);

      Command::ErrorReporter Reporter(
          llvm::SMLoc::getFromPointer(CommandName.data()), SrcMgr);
      Cmd->run(OS, Args, StdHeader, Records, Reporter);
    } else if (!Line.starts_with(CommentPrefix)) {
      // There is no comment or command on this line so we just write it as is.
      OS << P.first << "\n";
    }

    if (P.second.empty())
      break;
  }
}

void Generator::generateDecls(llvm::raw_ostream &OS,
                              const llvm::RecordKeeper &Records) {

  OS << "//===-- C standard declarations for " << StdHeader << " "
     << std::string(80 - (42 + StdHeader.size()), '-') << "===//\n"
     << "//\n"
     << "// Part of the LLVM Project, under the Apache License v2.0 with LLVM "
        "Exceptions.\n"
     << "// See https://llvm.org/LICENSE.txt for license information.\n"
     << "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
     << "//\n"
     << "//"
        "===-------------------------------------------------------------------"
        "---===//\n\n";

  std::string HeaderGuard(StdHeader.size(), '\0');
  llvm::transform(StdHeader, HeaderGuard.begin(), [](const char C) -> char {
    return !isalnum(C) ? '_' : llvm::toUpper(C);
  });
  OS << "#ifndef __LLVM_LIBC_DECLARATIONS_" << HeaderGuard << "\n"
     << "#define __LLVM_LIBC_DECLARATIONS_" << HeaderGuard << "\n\n";

  OS << "#ifndef __LIBC_ATTRS\n"
     << "#define __LIBC_ATTRS\n"
     << "#endif\n\n";

  OS << "#ifdef __cplusplus\n"
     << "extern \"C\" {\n"
     << "#endif\n\n";

  APIIndexer G(StdHeader, Records);
  for (auto &Name : EntrypointNameList) {
    // Filter out functions not exported by this header.
    if (G.FunctionSpecMap.find(Name) == G.FunctionSpecMap.end())
      continue;

    const llvm::Record *FunctionSpec = G.FunctionSpecMap[Name];
    const llvm::Record *RetValSpec = FunctionSpec->getValueAsDef("Return");
    const llvm::Record *ReturnType = RetValSpec->getValueAsDef("ReturnType");

    OS << G.getTypeAsString(ReturnType) << " " << Name << "(";

    auto ArgsList = FunctionSpec->getValueAsListOfDefs("Args");
    for (size_t i = 0; i < ArgsList.size(); ++i) {
      const llvm::Record *ArgType = ArgsList[i]->getValueAsDef("ArgType");
      OS << G.getTypeAsString(ArgType);
      if (i < ArgsList.size() - 1)
        OS << ", ";
    }

    OS << ") __LIBC_ATTRS;\n\n";
  }

  // Make another pass over entrypoints to emit object declarations.
  for (const auto &Name : EntrypointNameList) {
    if (G.ObjectSpecMap.find(Name) == G.ObjectSpecMap.end())
      continue;
    const llvm::Record *ObjectSpec = G.ObjectSpecMap[Name];
    auto Type = ObjectSpec->getValueAsString("Type");
    OS << "extern " << Type << " " << Name << " __LIBC_ATTRS;\n";
  }

  // Emit a final newline if we emitted any object declarations.
  if (llvm::any_of(EntrypointNameList, [&](const std::string &Name) {
        return G.ObjectSpecMap.find(Name) != G.ObjectSpecMap.end();
      }))
    OS << "\n";

  OS << "#ifdef __cplusplus\n"
     << "}\n"
     << "#endif\n\n";
  OS << "#endif\n";
}

} // namespace llvm_libc
