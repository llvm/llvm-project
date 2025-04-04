//===-- SystemLifetimeManagerDbg.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INITIALIZATION_SYSTEMLIFETIMEMANAGERDBG_H
#define LLDB_INITIALIZATION_SYSTEMLIFETIMEMANAGERDBG_H

#include "SystemLifetimeManager.h"
#include "lldb/Core/Debugger.h"

namespace lldb_private {

class SystemLifetimeManagerDbg : public SystemLifetimeManager {
public:
  SystemLifetimeManagerDbg() : SystemLifetimeManager() {};

private:
  virtual void
  InitializeDebugger(LoadPluginCallbackType plugin_callback) override {
    Debugger::Initialize(plugin_callback);
  };

  virtual void TerminateDebugger() override { Debugger::Terminate(); };

  // Noncopyable.
  SystemLifetimeManagerDbg(const SystemLifetimeManagerDbg &other) = delete;
  SystemLifetimeManagerDbg &
  operator=(const SystemLifetimeManagerDbg &other) = delete;
};
} // namespace lldb_private

#endif
