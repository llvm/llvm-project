//===----------- Implementation of IncludeFileCommand -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeFileCommand.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <cstdlib>

namespace llvm_libc {

const char IncludeFileCommand::Name[] = "include_file";

void IncludeFileCommand::run(llvm::raw_ostream &OS, const ArgVector &Args,
                             llvm::StringRef StdHeader,
                             llvm::RecordKeeper &Records,
                             const Command::ErrorReporter &Reporter) const {
  if (Args.size() != 1) {
    Reporter.printFatalError(
        "%%include_file command takes exactly 1 argument.");
  }

  llvm::StringRef IncludeFile = Args[0];
  auto Buffer = llvm::MemoryBuffer::getFileAsStream(IncludeFile);
  if (!Buffer)
    Reporter.printFatalError(llvm::StringRef("Unable to open ") + IncludeFile);

  llvm::StringRef Content = Buffer.get()->getBuffer();

  // If the included file has %%begin() command listed, then we want to write
  // only the content after the begin command.
  // TODO: The way the content is split below does not allow space within the
  // the parentheses and, before and after the command. This probably is too
  // strict and should be relaxed.
  auto P = Content.split("\n%%begin()\n");
  if (P.second.empty()) {
    // There was no %%begin in the content.
    OS << P.first;
  } else {
    OS << P.second;
  }
}

} // namespace llvm_libc
