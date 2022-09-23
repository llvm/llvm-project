//===-- Client.cpp - Remote Client ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A null implementation of the remote client, to be used when the actual
// implementation has been disabled during configuration.
//
//===----------------------------------------------------------------------===//

#include "../RemoteCache/Client.h"

using namespace clang;
using namespace clang::remote_cache;
using namespace llvm;

void AsyncCallerContext::anchor() {}
void KeyValueDBClient::anchor() {}
void CASDBClient::anchor() {}

Expected<ClientServices>
remote_cache::createCompilationCachingRemoteClient(StringRef SocketPath) {
  return createStringError(std::errc::not_supported,
                           "clang without remote caching support");
}
