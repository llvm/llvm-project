//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBPROTOCOLSERVER_H
#define LLDB_API_SBPROTOCOLSERVER_H

#include "lldb/API/SBDefines.h"

namespace lldb_private {
class SBProtocolServerImpl;
}

namespace lldb {

/// A server that speaks a debugging protocol.
///
/// Protocol servers are shared across debuggers. Creating a server for a
/// protocol that already has one hands back the existing server.
class LLDB_API SBProtocolServer {
public:
  SBProtocolServer();
  SBProtocolServer(const SBProtocolServer &rhs);
  SBProtocolServer &operator=(const SBProtocolServer &rhs);
  ~SBProtocolServer();

  /// Returns the protocol server for \p protocol, creating it if this process
  /// does not already have one. On failure \p error is set.
  static SBProtocolServer Create(const char *protocol, lldb::SBError &error);

  explicit operator bool() const;
  bool IsValid() const;

  /// Starts listening for clients on \p connection_uri, which must be an
  /// accepting URI such as "listen://[host]:port" or "accept:///path".
  lldb::SBError Start(const char *connection_uri);

  lldb::SBError Stop();

  /// Returns the URI clients can connect to, valid after a successful Start,
  /// or nullptr if the server is not listening.
  const char *GetConnectionURI();

private:
  std::unique_ptr<lldb_private::SBProtocolServerImpl> m_opaque_up;
};

} // namespace lldb

#endif
