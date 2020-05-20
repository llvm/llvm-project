//===--- Client.h - Connect to a remote index via gRPC -----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REMOTE_INDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REMOTE_INDEX_H

#include "index/Index.h"

namespace clang {
namespace clangd {
namespace remote {

/// Returns an SymbolIndex client that passes requests to remote index located
/// at \p Address. The client allows synchronous RPC calls.
///
/// This method attempts to resolve the address and establish the connection.
///
/// \returns nullptr if the address is not resolved during the function call or
/// if the project was compiled without Remote Index support.
std::unique_ptr<clangd::SymbolIndex> getClient(llvm::StringRef Address);

} // namespace remote
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REMOTE_INDEX_H
