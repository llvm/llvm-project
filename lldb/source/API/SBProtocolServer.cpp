//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBProtocolServer.h"
#include "lldb/API/SBError.h"
#include "lldb/Core/ProtocolServer.h"
#include "lldb/Host/Socket.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Instrumentation.h"
#include "lldb/Utility/UriParser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

/// PIMPL backing SBProtocolServer.
class lldb_private::SBProtocolServerImpl {
public:
  ProtocolServer *server = nullptr;
};

SBProtocolServer::SBProtocolServer()
    : m_opaque_up(std::make_unique<SBProtocolServerImpl>()) {
  LLDB_INSTRUMENT_VA(this);
}

SBProtocolServer::SBProtocolServer(const SBProtocolServer &rhs)
    : m_opaque_up(std::make_unique<SBProtocolServerImpl>(*rhs.m_opaque_up)) {
  LLDB_INSTRUMENT_VA(this, rhs);
}

SBProtocolServer &SBProtocolServer::operator=(const SBProtocolServer &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs)
    *m_opaque_up = *rhs.m_opaque_up;
  return *this;
}

SBProtocolServer::~SBProtocolServer() = default;

SBProtocolServer SBProtocolServer::Create(const char *protocol,
                                          SBError &error) {
  LLDB_INSTRUMENT_VA(protocol, error);

  SBProtocolServer server;
  error.Clear();

  if (!protocol || !protocol[0]) {
    error.SetErrorString("no protocol specified");
    return server;
  }

  ProtocolServer *protocol_server = ProtocolServer::GetOrCreate(protocol);
  if (!protocol_server) {
    error.SetErrorStringWithFormat(
        "unsupported protocol: %s. Supported protocols are: %s", protocol,
        llvm::join(ProtocolServer::GetSupportedProtocols(), ", ").c_str());
    return server;
  }

  server.m_opaque_up->server = protocol_server;
  return server;
}

SBProtocolServer::operator bool() const {
  LLDB_INSTRUMENT_VA(this);

  return m_opaque_up->server != nullptr;
}

bool SBProtocolServer::IsValid() const {
  LLDB_INSTRUMENT_VA(this);

  return m_opaque_up->server != nullptr;
}

SBError SBProtocolServer::Start(const char *connection_uri) {
  LLDB_INSTRUMENT_VA(this, connection_uri);

  SBError error;
  if (!m_opaque_up->server) {
    error.SetErrorString("invalid protocol server");
    return error;
  }

  const char *connection_error =
      "unsupported connection specifier, expected 'accept:///path' or "
      "'listen://[host]:port'";
  std::optional<URI> uri = URI::Parse(connection_uri ? connection_uri : "");
  if (!uri) {
    error.SetErrorString(connection_error);
    return error;
  }

  std::optional<Socket::ProtocolModePair> protocol_and_mode =
      Socket::GetProtocolAndMode(uri->scheme);
  if (!protocol_and_mode || protocol_and_mode->second != Socket::ModeAccept) {
    error.SetErrorString(connection_error);
    return error;
  }

  ProtocolServer::Connection connection;
  connection.protocol = protocol_and_mode->first;
  if (connection.protocol == Socket::SocketProtocol::ProtocolUnixDomain)
    connection.name = uri->path;
  else
    connection.name =
        llvm::formatv("[{0}]:{1}",
                      uri->hostname.empty() ? "0.0.0.0" : uri->hostname,
                      uri->port.value_or(0))
            .str();

  if (llvm::Error err = m_opaque_up->server->Start(connection))
    error.SetErrorString(llvm::toString(std::move(err)).c_str());

  return error;
}

SBError SBProtocolServer::Stop() {
  LLDB_INSTRUMENT_VA(this);

  SBError error;
  if (!m_opaque_up->server) {
    error.SetErrorString("invalid protocol server");
    return error;
  }

  if (llvm::Error err = m_opaque_up->server->Stop())
    error.SetErrorString(llvm::toString(std::move(err)).c_str());

  return error;
}

const char *SBProtocolServer::GetConnectionURI() {
  LLDB_INSTRUMENT_VA(this);

  if (!m_opaque_up->server)
    return nullptr;

  Socket *socket = m_opaque_up->server->GetSocket();
  if (!socket)
    return nullptr;

  std::vector<std::string> uris = socket->GetListeningConnectionURI();
  if (uris.empty())
    return nullptr;

  // A listening socket may report several equivalent URIs (e.g. IPv6 and
  // IPv4 loopback). Return the first, interned so the pointer stays valid.
  return ConstString(uris.front()).GetCString();
}
