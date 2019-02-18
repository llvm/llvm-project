//===-- ClangdFuzzer.cpp - Fuzz clangd ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a function that runs clangd on a single input.
/// This function is then linked into the Fuzzer library.
///
//===----------------------------------------------------------------------===//

#include "ClangdLSPServer.h"
#include "ClangdServer.h"
#include "CodeComplete.h"
#include <sstream>
#include <stdio.h>

using namespace clang::clangd;

extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  if (size == 0)
    return 0;

  // fmemopen isn't portable, but I think we only run the fuzzer on Linux.
  std::FILE *In = fmemopen(data, size, "r");
  auto Transport = newJSONTransport(In, llvm::nulls(),
                                    /*InMirror=*/nullptr, /*Pretty=*/false,
                                    /*Style=*/JSONStreamStyle::Standard);
  CodeCompleteOptions CCOpts;
  CCOpts.EnableSnippets = false;
  ClangdServer::Options Opts;

  // Initialize and run ClangdLSPServer.
  ClangdLSPServer LSPServer(*Transport, CCOpts, llvm::None, false, Opts);
  LSPServer.run();
  return 0;
}
