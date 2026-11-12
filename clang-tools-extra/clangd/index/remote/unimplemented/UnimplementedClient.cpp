//===--- UnimplementedClient.cpp ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/remote/Client.h"
#include "support/Logger.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace clangd {
namespace remote {

std::unique_ptr<clangd::SymbolIndex> getClient(llvm::StringRef Address,
                                               llvm::StringRef IndexRoot) {
  elog("Can't create SymbolIndex client without Remote Index support.");
  return nullptr;
}

} // namespace remote
} // namespace clangd
} // namespace clang
