//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_POSIX_DOMAINSOCKETPOSIX_H
#define LLDB_HOST_POSIX_DOMAINSOCKETPOSIX_H

#include "lldb/Host/common/DomainSocket.h"

namespace lldb_private {

/// \class DomainSocketPosix DomainSocketPosix.h
/// "lldb/Host/posix/DomainSocketPosix.h"
/// POSIX implementation of the platform-specific parts of DomainSocket.
class DomainSocketPosix : public DomainSocket {
public:
  using DomainSocket::DomainSocket;

  /// Create a connected pair of domain sockets using socketpair(2).
  static llvm::Expected<Pair> CreatePair();
};

} // namespace lldb_private

#endif // LLDB_HOST_POSIX_DOMAINSOCKETPOSIX_H
