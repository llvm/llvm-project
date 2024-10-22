//===-- The main header generation class ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_HDRGEN_GENERATOR_H
#define LLVM_LIBC_UTILS_HDRGEN_GENERATOR_H

#include "Command.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace llvm {

class raw_ostream;
class RecordKeeper;

} // namespace llvm

namespace llvm_libc {

class Command;

class Generator {
  llvm::StringRef HeaderDefFile;
  const std::vector<std::string> &EntrypointNameList;
  llvm::StringRef StdHeader;
  std::unordered_map<std::string, std::string> &ArgMap;

  std::unique_ptr<Command> IncludeFileCmd;
  std::unique_ptr<Command> PublicAPICmd;

  Command *getCommandHandler(llvm::StringRef CommandName);

  void parseCommandArgs(llvm::StringRef ArgStr, ArgVector &Args);

  void printError(llvm::StringRef Msg);

public:
  Generator(const std::string &DefFile, const std::vector<std::string> &EN,
            const std::string &Header,
            std::unordered_map<std::string, std::string> &Map)
      : HeaderDefFile(DefFile), EntrypointNameList(EN), StdHeader(Header),
        ArgMap(Map) {}

  void generate(llvm::raw_ostream &OS, const llvm::RecordKeeper &Records);
  void generateDecls(llvm::raw_ostream &OS, const llvm::RecordKeeper &Records);
};

} // namespace llvm_libc

#endif // LLVM_LIBC_UTILS_HDRGEN_GENERATOR_H
