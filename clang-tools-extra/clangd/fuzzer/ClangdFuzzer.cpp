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

extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  if (size == 0)
    return 0;

  clang::clangd::JSONOutput Out(llvm::nulls(), llvm::nulls(),
                                clang::clangd::Logger::Error, nullptr);
  clang::clangd::CodeCompleteOptions CCOpts;
  CCOpts.EnableSnippets = false;
  clang::clangd::ClangdServer::Options Opts;

  // Initialize and run ClangdLSPServer.
  clang::clangd::ClangdLSPServer LSPServer(Out, CCOpts, llvm::None, false,
                                           Opts);
  // fmemopen isn't portable, but I think we only run the fuzzer on Linux.
  LSPServer.run(fmemopen(data, size, "r"));
  return 0;
}
