//===-- ProtocolServer.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_PROTOCOLSERVER_H
#define LLDB_CORE_PROTOCOLSERVER_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/Host/Socket.h"
#include "lldb/lldb-private-interfaces.h"

namespace lldb_private {

/// \class ProtocolServer ProtocolServer.h "lldb/Core/ProtocolServer.h"
/// A plug-in interface definition class for protocol servers.
///
/// Protocol servers provide network-based debugging interfaces that allow
/// external tools and IDEs to communicate with LLDB. Each plugin implements
/// a specific protocol (e.g., GDB remote protocol, DAP, MCP) and handles
/// connection management, packet parsing, and command dispatch.
///
/// Protocol servers are singleton instances managed by LLDB. They are created
/// on-demand when first requested via GetOrCreate() and persist until
/// explicitly terminated. Multiple clients can connect to the same protocol
/// server instance if the underlying protocol supports it.
///
/// Plugin Selection and Instantiation:
/// Protocol server plugins are selected by name when calling GetOrCreate().
/// LLDB maintains a registry of available protocol server plugins, and
/// GetOrCreate() looks up the appropriate CreateInstance callback for the
/// requested protocol name. If a server instance already exists for that
/// protocol, the existing instance is returned rather than creating a new one.
///
/// Thread Safety:
/// Implementations should be thread-safe, as Start() and Stop() may be called
/// from different threads. The base class GetOrCreate() method uses internal
/// locking to ensure thread-safe singleton creation.
///
/// Lifecycle:
/// Protocol servers are typically started during debugger initialization or
/// when a user explicitly enables a protocol. They continue running until
/// the debugger terminates or until Terminate() is called, which stops all
/// active protocol server instances.
class ProtocolServer : public PluginInterface {
public:
  ProtocolServer() = default;
  virtual ~ProtocolServer() = default;

  static ProtocolServer *GetOrCreate(llvm::StringRef name);

  static llvm::Error Terminate();

  static std::vector<llvm::StringRef> GetSupportedProtocols();

  struct Connection {
    Socket::SocketProtocol protocol;
    std::string name;
  };

  virtual llvm::Error Start(Connection connection) = 0;
  virtual llvm::Error Stop() = 0;

  virtual Socket *GetSocket() const = 0;
};

} // namespace lldb_private

#endif
