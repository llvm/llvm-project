//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_WINDOWS_DOMAINSOCKETWINDOWS_H
#define LLDB_HOST_WINDOWS_DOMAINSOCKETWINDOWS_H

#include "lldb/Host/common/DomainSocket.h"

namespace lldb_private {

/// \class DomainSocketWindows DomainSocketWindows.h
/// "lldb/Host/windows/DomainSocketWindows.h"
/// Windows implementation of the platform-specific parts of DomainSocket.
class DomainSocketWindows : public DomainSocket {
public:
  using DomainSocket::DomainSocket;

  /// Create a connected pair of domain sockets. Windows has no socketpair(2),
  /// so this is emulated with a transient listening socket.
  static llvm::Expected<Pair> CreatePair();
};

} // namespace lldb_private

#endif // LLDB_HOST_WINDOWS_DOMAINSOCKETWINDOWS_H
