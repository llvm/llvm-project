//===-- Base class for header generation commands ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_HDRGEN_COMMAND_H
#define LLVM_LIBC_UTILS_HDRGEN_COMMAND_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"

#include <cstdlib>

namespace llvm {

class raw_ostream;
class RecordKeeper;

} // namespace llvm

namespace llvm_libc {

typedef llvm::SmallVector<llvm::StringRef, 4> ArgVector;

class Command {
public:
  class ErrorReporter {
    llvm::SMLoc Loc;
    const llvm::SourceMgr &SrcMgr;

  public:
    ErrorReporter(llvm::SMLoc L, llvm::SourceMgr &SM) : Loc(L), SrcMgr(SM) {}

    [[noreturn]] void printFatalError(llvm::Twine Msg) const {
      SrcMgr.PrintMessage(Loc, llvm::SourceMgr::DK_Error, Msg);
      std::exit(1);
    }
  };

  virtual ~Command();

  virtual void run(llvm::raw_ostream &OS, const ArgVector &Args,
                   llvm::StringRef StdHeader, const llvm::RecordKeeper &Records,
                   const ErrorReporter &Reporter) const = 0;
};

} // namespace llvm_libc

#endif // LLVM_LIBC_UTILS_HDRGEN_COMMAND_H
