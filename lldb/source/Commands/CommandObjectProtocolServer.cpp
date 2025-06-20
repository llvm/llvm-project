//===-- CommandObjectProtocolServer.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectProtocolServer.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/ProtocolServer.h"
#include "lldb/Host/Socket.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/UriParser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatAdapters.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_mcp
#include "CommandOptions.inc"

static std::vector<llvm::StringRef> GetSupportedProtocols() {
  std::vector<llvm::StringRef> supported_protocols;
  size_t i = 0;

  for (llvm::StringRef protocol_name =
           PluginManager::GetProtocolServerPluginNameAtIndex(i++);
       !protocol_name.empty();
       protocol_name = PluginManager::GetProtocolServerPluginNameAtIndex(i++)) {
    supported_protocols.push_back(protocol_name);
  }

  return supported_protocols;
}

class CommandObjectProtocolServerStart : public CommandObjectParsed {
public:
  CommandObjectProtocolServerStart(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "protocol-server start",
                            "start protocol server",
                            "protocol-server start <protocol> <connection>") {
    AddSimpleArgumentList(lldb::eArgTypeProtocol, eArgRepeatPlain);
    AddSimpleArgumentList(lldb::eArgTypeConnectURL, eArgRepeatPlain);
  }

  ~CommandObjectProtocolServerStart() override = default;

protected:
  void DoExecute(Args &args, CommandReturnObject &result) override {
    if (args.GetArgumentCount() < 1) {
      result.AppendError("no protocol specified");
      return;
    }

    llvm::StringRef protocol = args.GetArgumentAtIndex(0);
    std::vector<llvm::StringRef> supported_protocols = GetSupportedProtocols();
    if (llvm::find(supported_protocols, protocol) ==
        supported_protocols.end()) {
      result.AppendErrorWithFormatv(
          "unsupported protocol: {0}. Supported protocols are: {1}", protocol,
          llvm::join(GetSupportedProtocols(), ", "));
      return;
    }

    if (args.GetArgumentCount() < 2) {
      result.AppendError("no connection specified");
      return;
    }
    llvm::StringRef connection_uri = args.GetArgumentAtIndex(1);

    ProtocolServerSP server_sp = GetDebugger().GetProtocolServer(protocol);
    if (!server_sp)
      server_sp = ProtocolServer::Create(protocol, GetDebugger());

    const char *connection_error =
        "unsupported connection specifier, expected 'accept:///path' or "
        "'listen://[host]:port', got '{0}'.";
    auto uri = lldb_private::URI::Parse(connection_uri);
    if (!uri) {
      result.AppendErrorWithFormatv(connection_error, connection_uri);
      return;
    }

    std::optional<Socket::ProtocolModePair> protocol_and_mode =
        Socket::GetProtocolAndMode(uri->scheme);
    if (!protocol_and_mode || protocol_and_mode->second != Socket::ModeAccept) {
      result.AppendErrorWithFormatv(connection_error, connection_uri);
      return;
    }

    ProtocolServer::Connection connection;
    connection.protocol = protocol_and_mode->first;
    connection.name =
        formatv("[{0}]:{1}", uri->hostname.empty() ? "0.0.0.0" : uri->hostname,
                uri->port.value_or(0));

    if (llvm::Error error = server_sp->Start(connection)) {
      result.AppendErrorWithFormatv("{0}", llvm::fmt_consume(std::move(error)));
      return;
    }

    GetDebugger().AddProtocolServer(server_sp);

    if (Socket *socket = server_sp->GetSocket()) {
      std::string address =
          llvm::join(socket->GetListeningConnectionURI(), ", ");
      result.AppendMessageWithFormatv(
          "{0} server started with connection listeners: {1}", protocol,
          address);
    }
  }
};

class CommandObjectProtocolServerStop : public CommandObjectParsed {
public:
  CommandObjectProtocolServerStop(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "protocol-server stop",
                            "stop protocol server",
                            "protocol-server stop <protocol>") {
    AddSimpleArgumentList(lldb::eArgTypeProtocol, eArgRepeatPlain);
  }

  ~CommandObjectProtocolServerStop() override = default;

protected:
  void DoExecute(Args &args, CommandReturnObject &result) override {
    if (args.GetArgumentCount() < 1) {
      result.AppendError("no protocol specified");
      return;
    }

    llvm::StringRef protocol = args.GetArgumentAtIndex(0);
    std::vector<llvm::StringRef> supported_protocols = GetSupportedProtocols();
    if (llvm::find(supported_protocols, protocol) ==
        supported_protocols.end()) {
      result.AppendErrorWithFormatv(
          "unsupported protocol: {0}. Supported protocols are: {1}", protocol,
          llvm::join(GetSupportedProtocols(), ", "));
      return;
    }

    Debugger &debugger = GetDebugger();

    ProtocolServerSP server_sp = debugger.GetProtocolServer(protocol);
    if (!server_sp) {
      result.AppendError(
          llvm::formatv("no {0} protocol server running", protocol).str());
      return;
    }

    if (llvm::Error error = server_sp->Stop()) {
      result.AppendErrorWithFormatv("{0}", llvm::fmt_consume(std::move(error)));
      return;
    }

    debugger.RemoveProtocolServer(server_sp);
  }
};

CommandObjectProtocolServer::CommandObjectProtocolServer(
    CommandInterpreter &interpreter)
    : CommandObjectMultiword(interpreter, "protocol-server",
                             "Start and stop a protocol server.",
                             "protocol-server") {
  LoadSubCommand("start", CommandObjectSP(new CommandObjectProtocolServerStart(
                              interpreter)));
  LoadSubCommand("stop", CommandObjectSP(
                             new CommandObjectProtocolServerStop(interpreter)));
}

CommandObjectProtocolServer::~CommandObjectProtocolServer() = default;
