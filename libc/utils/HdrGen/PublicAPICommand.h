//===-- Implementation of PublicAPICommand ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_HDRGEN_PUBLICAPICOMMAND_H
#define LLVM_LIBC_UTILS_HDRGEN_PUBLICAPICOMMAND_H

#include "Command.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

namespace llvm {

class raw_ostream;
class Record;
class RecordKeeper;

} // namespace llvm

namespace llvm_libc {

class PublicAPICommand : public Command {
private:
  const std::vector<std::string> &EntrypointNameList;

public:
  static const char Name[];

  PublicAPICommand(const std::vector<std::string> &EntrypointNames)
      : EntrypointNameList(EntrypointNames) {}

  void run(llvm::raw_ostream &OS, const ArgVector &Args,
           llvm::StringRef StdHeader, llvm::RecordKeeper &Records,
           const Command::ErrorReporter &Reporter) const override;
};

} // namespace llvm_libc

#endif // LLVM_LIBC_UTILS_HDRGEN_PUBLICAPICOMMAND_H
