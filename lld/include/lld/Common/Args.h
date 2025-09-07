//===- Args.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ARGS_H
#define LLD_ARGS_H

#include "lld/Common/LLVM.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/MemoryBuffer.h"
#include <vector>

namespace llvm {
class StringSaver;
namespace opt {
class Arg;
class InputArgList;
class OptSpecifier;
class OptTable;
} // namespace opt
} // namespace llvm

namespace lld {
namespace args {

int getCGOptLevel(int optLevelLTO);

int64_t getInteger(llvm::opt::InputArgList &args, unsigned key,
                   int64_t Default);

int64_t getHex(llvm::opt::InputArgList &args, unsigned key, int64_t Default);

llvm::SmallVector<StringRef, 0> getStrings(llvm::opt::InputArgList &args,
                                           int id);

uint64_t getZOptionValue(llvm::opt::InputArgList &args, int id, StringRef key,
                         uint64_t Default);

std::vector<StringRef> getLines(MemoryBufferRef mb);

StringRef getFilenameWithoutExe(StringRef path);

StringRef getOptionSpellingLikeArg(llvm::opt::OptTable &optTable,
                                   llvm::opt::OptSpecifier opt,
                                   llvm::opt::Arg *arg,
                                   llvm::StringSaver &saver);

} // namespace args
} // namespace lld

#endif
