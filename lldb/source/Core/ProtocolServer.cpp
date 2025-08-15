//===-- ProtocolServer.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ProtocolServer.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb_private;
using namespace lldb;

ProtocolServer *ProtocolServer::GetOrCreate(llvm::StringRef name) {
  static std::mutex g_mutex;
  static llvm::StringMap<ProtocolServerUP> g_protocol_server_instances;

  std::lock_guard<std::mutex> guard(g_mutex);

  auto it = g_protocol_server_instances.find(name);
  if (it != g_protocol_server_instances.end())
    return it->second.get();

  if (ProtocolServerCreateInstance create_callback =
          PluginManager::GetProtocolCreateCallbackForPluginName(name)) {
    auto pair =
        g_protocol_server_instances.try_emplace(name, create_callback());
    return pair.first->second.get();
  }

  return nullptr;
}

std::vector<llvm::StringRef> ProtocolServer::GetSupportedProtocols() {
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
