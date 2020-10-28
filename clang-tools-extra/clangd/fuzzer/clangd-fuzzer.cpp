//===-- ClangdFuzzer.cpp - Fuzz clangd ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a function that runs clangd on a single input.
/// This function is then linked into the Fuzzer library.
///
//===----------------------------------------------------------------------===//

#include "ClangdLSPServer.h"
#include "ClangdServer.h"
#include "support/ThreadsafeFS.h"
#include <cstdio>
#include <sstream>

using namespace clang::clangd;

extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  if (size == 0)
    return 0;

  // fmemopen isn't portable, but I think we only run the fuzzer on Linux.
  std::FILE *In = fmemopen(data, size, "r");
  auto Transport = newJSONTransport(In, llvm::nulls(),
                                    /*InMirror=*/nullptr, /*Pretty=*/false,
                                    /*Style=*/JSONStreamStyle::Delimited);
  RealThreadsafeFS FS;
  CodeCompleteOptions CCOpts;
  ClangdLSPServer::Options Opts;
  Opts.CodeComplete.EnableSnippets = false;
  Opts.UseDirBasedCDB = false;

  // Initialize and run ClangdLSPServer.
  ClangdLSPServer LSPServer(*Transport, FS, Opts);
  LSPServer.run();
  return 0;
}
