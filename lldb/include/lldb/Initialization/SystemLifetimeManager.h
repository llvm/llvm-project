//===-- SystemLifetimeManager.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INITIALIZATION_SYSTEM_LIFETIME_MANAGER_H
#define LLDB_INITIALIZATION_SYSTEM_LIFETIME_MANAGER_H

#include "lldb/Host/Mutex.h"
#include "lldb/lldb-private-types.h"

#include <memory>

namespace lldb_private {
class SystemInitializer;

class SystemLifetimeManager {
public:
  SystemLifetimeManager();
  ~SystemLifetimeManager();

  void Initialize(std::unique_ptr<SystemInitializer> initializer,
                  LoadPluginCallbackType plugin_callback);
  void Terminate();

private:
  Mutex m_mutex;
  std::unique_ptr<SystemInitializer> m_initializer;
  bool m_initialized;

  // Noncopyable.
  SystemLifetimeManager(const SystemLifetimeManager &other) = delete;
  SystemLifetimeManager &operator=(const SystemLifetimeManager &other) = delete;
};
}

#endif
