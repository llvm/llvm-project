//===-- Class which implements the %%include_file command -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_HDRGEN_INCLUDE_COMMAND_H
#define LLVM_LIBC_UTILS_HDRGEN_INCLUDE_COMMAND_H

#include "Command.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace llvm_libc {

class IncludeFileCommand : public Command {
public:
  static const char Name[];

  void run(llvm::raw_ostream &OS, const ArgVector &Args,
           llvm::StringRef StdHeader, const llvm::RecordKeeper &Records,
           const Command::ErrorReporter &Reporter) const override;
};

} // namespace llvm_libc

#endif // LLVM_LIBC_UTILS_HDRGEN_INCLUDE_COMMAND_H
