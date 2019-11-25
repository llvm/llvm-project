//===---------- Implementation of PublicAPICommand --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Command.h"

#include "llvm/ADT/StringRef.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace llvm {

class raw_ostream;
class Record;
class RecordKeeper;

} // namespace llvm

namespace llvm_libc {

class PublicAPICommand : public Command {
public:
  static const char Name[];

  void run(llvm::raw_ostream &OS, const ArgVector &Args,
           llvm::StringRef StdHeader, llvm::RecordKeeper &Records,
           const Command::ErrorReporter &Reporter) const override;
};

} // namespace llvm_libc
