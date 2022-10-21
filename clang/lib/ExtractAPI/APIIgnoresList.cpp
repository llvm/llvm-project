//===- ExtractAPI/APIIgnoresList.cpp -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements APIIgnoresList that allows users to specifiy a file
/// containing symbols to ignore during API extraction.
///
//===----------------------------------------------------------------------===//

#include "clang/ExtractAPI/APIIgnoresList.h"
#include "clang/Basic/FileManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

using namespace clang;
using namespace clang::extractapi;
using namespace llvm;

char IgnoresFileNotFound::ID;

void IgnoresFileNotFound::log(llvm::raw_ostream &os) const {
  os << "Could not find API ignores file " << Path;
}

std::error_code IgnoresFileNotFound::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

Expected<APIIgnoresList> APIIgnoresList::create(StringRef IgnoresFilePath,
                                                FileManager &FM) {
  auto BufferOrErr = FM.getBufferForFile(IgnoresFilePath);
  if (!BufferOrErr)
    return make_error<IgnoresFileNotFound>(IgnoresFilePath);

  auto Buffer = std::move(BufferOrErr.get());
  SmallVector<StringRef, 32> Lines;
  Buffer->getBuffer().split(Lines, '\n', /*MaxSplit*/ -1, /*KeepEmpty*/ false);
  // Symbol names don't have spaces in them, let's just remove these in case the
  // input is slighlty malformed.
  transform(Lines, Lines.begin(), [](StringRef Line) { return Line.trim(); });
  sort(Lines);
  return APIIgnoresList(std::move(Lines), std::move(Buffer));
}

bool APIIgnoresList::shouldIgnore(StringRef SymbolName) const {
  auto It = lower_bound(SymbolsToIgnore, SymbolName);
  return (It != SymbolsToIgnore.end()) && (*It == SymbolName);
}
