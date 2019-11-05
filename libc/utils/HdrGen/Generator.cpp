//===---- Implementation of the main header generation class -----*- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Generator.h"

#include "IncludeFileCommand.h"
#include "PublicAPICommand.h"

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
      PublicAPICmd = std::make_unique<PublicAPICommand>();
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
    if (A.startswith(ParamNamePrefix) && A.endswith(ParamNameSuffix)) {
      A = A.drop_front(ParamNamePrefixSize).drop_back(ParamNameSuffixSize);
      A = ArgMap[A];
    }
  }
}

void Generator::generate(llvm::raw_ostream &OS, llvm::RecordKeeper &Records) {
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
    if (Line.startswith(CommandPrefix)) {
      Line = Line.drop_front(CommandPrefixSize);

      P = Line.split("(");
      if (P.second.empty() || P.second[P.second.size() - 1] != ')') {
        SrcMgr.PrintMessage(llvm::SMLoc::getFromPointer(P.second.data()),
                            llvm::SourceMgr::DK_Error,
                            "Command argument list should begin with '(' "
                            "and end with ')'.");
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
    } else if (!Line.startswith(CommentPrefix)) {
      // There is no comment or command on this line so we just write it as is.
      OS << P.first << "\n";
    }

    if (P.second.empty())
      break;
  }
}

} // namespace llvm_libc
