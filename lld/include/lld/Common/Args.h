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
namespace opt {
class InputArgList;
}
} // namespace llvm

namespace lld {
namespace args {

llvm::CodeGenOpt::Level getCGOptLevel(int OptLevelLTO);

int getInteger(llvm::opt::InputArgList &Args, unsigned Key, int Default);

std::vector<StringRef> getStrings(llvm::opt::InputArgList &Args, int Id);

uint64_t getZOptionValue(llvm::opt::InputArgList &Args, int Id, StringRef Key,
                         uint64_t Default);

std::vector<StringRef> getLines(MemoryBufferRef MB);

StringRef getFilenameWithoutExe(StringRef Path);

} // namespace args
} // namespace lld

#endif
