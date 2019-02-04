//===- Args.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Args.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace lld;

// TODO(sbc): Remove this once CGOptLevel can be set completely based on bitcode
// function metadata.
CodeGenOpt::Level lld::args::getCGOptLevel(int OptLevelLTO) {
  if (OptLevelLTO == 3)
    return CodeGenOpt::Aggressive;
  assert(OptLevelLTO < 3);
  return CodeGenOpt::Default;
}

int lld::args::getInteger(opt::InputArgList &Args, unsigned Key, int Default) {
  auto *A = Args.getLastArg(Key);
  if (!A)
    return Default;

  int V;
  if (to_integer(A->getValue(), V, 10))
    return V;

  StringRef Spelling = Args.getArgString(A->getIndex());
  error(Spelling + ": number expected, but got '" + A->getValue() + "'");
  return 0;
}

std::vector<StringRef> lld::args::getStrings(opt::InputArgList &Args, int Id) {
  std::vector<StringRef> V;
  for (auto *Arg : Args.filtered(Id))
    V.push_back(Arg->getValue());
  return V;
}

uint64_t lld::args::getZOptionValue(opt::InputArgList &Args, int Id,
                                    StringRef Key, uint64_t Default) {
  for (auto *Arg : Args.filtered_reverse(Id)) {
    std::pair<StringRef, StringRef> KV = StringRef(Arg->getValue()).split('=');
    if (KV.first == Key) {
      uint64_t Result = Default;
      if (!to_integer(KV.second, Result))
        error("invalid " + Key + ": " + KV.second);
      return Result;
    }
  }
  return Default;
}

std::vector<StringRef> lld::args::getLines(MemoryBufferRef MB) {
  SmallVector<StringRef, 0> Arr;
  MB.getBuffer().split(Arr, '\n');

  std::vector<StringRef> Ret;
  for (StringRef S : Arr) {
    S = S.trim();
    if (!S.empty() && S[0] != '#')
      Ret.push_back(S);
  }
  return Ret;
}

StringRef lld::args::getFilenameWithoutExe(StringRef Path) {
  if (Path.endswith_lower(".exe"))
    return sys::path::stem(Path);
  return sys::path::filename(Path);
}
