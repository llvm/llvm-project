//===-- SystemInitializerFull.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemInitializerFull.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/Host.h"
#include "lldb/Initialization/SystemInitializerCommon.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Target/ProcessTrace.h"
#include "lldb/Utility/Reproducer.h"
#include "lldb/Utility/Timer.h"
#include "llvm/Support/TargetSelect.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#include "llvm/ExecutionEngine/MCJIT.h"
#pragma clang diagnostic pop

#include <string>

#define LLDB_PLUGIN(p) LLDB_PLUGIN_DECLARE(p)
#include "Plugins/Plugins.def"

#if LLDB_ENABLE_PYTHON
#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"

constexpr lldb_private::HostInfo::SharedLibraryDirectoryHelper
    *g_shlib_dir_helper =
        lldb_private::ScriptInterpreterPython::SharedLibraryDirectoryHelper;

#else
constexpr lldb_private::HostInfo::SharedLibraryDirectoryHelper
    *g_shlib_dir_helper = 0;
#endif

using namespace lldb_private;

SystemInitializerFull::SystemInitializerFull()
    : SystemInitializerCommon(g_shlib_dir_helper) {}
SystemInitializerFull::~SystemInitializerFull() = default;

llvm::Error SystemInitializerFull::Initialize() {
  llvm::Error error = SystemInitializerCommon::Initialize();
  if (error) {
    // During active replay, the ::Initialize call is replayed like any other
    // SB API call and the return value is ignored. Since we can't intercept
    // this, we terminate here before the uninitialized debugger inevitably
    // crashes.
    if (repro::Reproducer::Instance().IsReplaying())
      llvm::report_fatal_error(std::move(error));
    return error;
  }

  // Initialize LLVM and Clang
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();

#define LLDB_PLUGIN(p) LLDB_PLUGIN_INITIALIZE(p);
#include "Plugins/Plugins.def"

  // Initialize plug-ins in core LLDB
  ProcessTrace::Initialize();

  // Scan for any system or user LLDB plug-ins
  PluginManager::Initialize();

  // The process settings need to know about installed plug-ins, so the
  // Settings must be initialized AFTER PluginManager::Initialize is called.
  Debugger::SettingsInitialize();

  return llvm::Error::success();
}

void SystemInitializerFull::Terminate() {
  Debugger::SettingsTerminate();

  // Terminate plug-ins in core LLDB
  ProcessTrace::Terminate();

  // Terminate and unload and loaded system or user LLDB plug-ins
  PluginManager::Terminate();

#define LLDB_PLUGIN(p) LLDB_PLUGIN_TERMINATE(p);
#include "Plugins/Plugins.def"

  // Now shutdown the common parts, in reverse order.
  SystemInitializerCommon::Terminate();
}
