//===--- LSPBinder.cpp - Tables of LSP handlers --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LSPBinder.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {

// Keep parseError out of every parse<T> instantiation.
LLVM_ATTRIBUTE_NOINLINE llvm::Error
LSPBinder::parseError(const llvm::json::Value &Raw, llvm::StringRef PayloadName,
                      llvm::StringRef PayloadKind,
                      const llvm::json::Path::Root &Root) {
  elog("Failed to decode {0} {1}: {2}", PayloadName, PayloadKind,
       Root.getError());
  // Dump the relevant parts of the broken message.
  std::string Context;
  llvm::raw_string_ostream OS(Context);
  Root.printErrorContext(Raw, OS);
  vlog("{0}", OS.str());
  // Report the error (e.g. to the client).
  return llvm::make_error<LSPError>(
      llvm::formatv("failed to decode {0} {1}: {2}", PayloadName, PayloadKind,
                    fmt_consume(Root.getError())),
      ErrorCode::InvalidParams);
}

} // namespace clangd
} // namespace clang
