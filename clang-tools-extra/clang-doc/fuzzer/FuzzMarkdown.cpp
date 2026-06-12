//===-- FuzzMarkdown.cpp - Fuzzer for the clang-doc Markdown parser -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a libFuzzer harness for parseMarkdown(). It feeds
/// arbitrary bytes to the parser and checks that it never crashes. The parsed
/// nodes are walked so the returned tree is exercised, not just allocated.
///
//===----------------------------------------------------------------------===//

#include "support/Markdown.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include <cstddef>
#include <cstdint>

using namespace clang::doc::markdown;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  llvm::BumpPtrAllocator Arena;
  llvm::StringRef Input(reinterpret_cast<const char *>(Data), Size);
  for (const MDNode *Node : parseMarkdown(Input, Arena))
    (void)Node->Kind;
  return 0;
}
