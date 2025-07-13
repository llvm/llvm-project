//===-- SystemInitializerFull.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemInitializerFull.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Progress.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/Host.h"
#include "lldb/Initialization/SystemInitializerCommon.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Target/ProcessTrace.h"
#include "lldb/Utility/Timer.h"
#include "lldb/Version/Version.h"
#include "llvm/Support/CommandLine.h"
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
    *g_shlib_dir_helper = nullptr;
#endif

using namespace lldb_private;

SystemInitializerFull::SystemInitializerFull()
    : SystemInitializerCommon(g_shlib_dir_helper) {}
SystemInitializerFull::~SystemInitializerFull() = default;

llvm::Error SystemInitializerFull::Initialize() {
  llvm::Error error = SystemInitializerCommon::Initialize();
  if (error)
    return error;

  // Initialize LLVM and Clang
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();

  // Initialize the command line parser in LLVM. This usually isn't necessary
  // as we aren't dealing with command line options here, but otherwise some
  // other code in Clang/LLVM might be tempted to call this function from a
  // different thread later on which won't work (as the function isn't
  // thread-safe).
  const char *arg0 = "lldb";
  llvm::cl::ParseCommandLineOptions(1, &arg0);

#define LLDB_PLUGIN(p) LLDB_PLUGIN_INITIALIZE(p);
#include "Plugins/Plugins.def"

  // Scan for any system or user LLDB plug-ins.
  PluginManager::Initialize();

  // The process settings need to know about installed plug-ins, so the
  // Settings must be initialized AFTER PluginManager::Initialize is called.
  Debugger::SettingsInitialize();

  // Use the Debugger's LLDBAssert callback.
  SetLLDBAssertCallback(Debugger::AssertCallback);

  // Use the system log to report errors that would otherwise get dropped.
  SetLLDBErrorLog(GetLog(SystemLog::System));

  LLDB_LOG(GetLog(SystemLog::System), "{0}", GetVersion());

  auto LoadPlugin = [](const lldb::DebuggerSP &debugger_sp,
                       const FileSpec &spec,
                       Status &error) -> llvm::sys::DynamicLibrary {
    llvm::sys::DynamicLibrary dynlib =
        llvm::sys::DynamicLibrary::getPermanentLibrary(spec.GetPath().c_str());
    if (dynlib.isValid()) {
      typedef bool (*LLDBCommandPluginInit)(lldb::SBDebugger debugger);

      lldb::SBDebugger debugger_sb(debugger_sp);
      // This calls the bool lldb::PluginInitialize(lldb::SBDebugger debugger)
      // function.
      // TODO: mangle this differently for your system - on OSX, the first
      // underscore needs to be removed and the second one stays
      LLDBCommandPluginInit init_func =
          (LLDBCommandPluginInit)(uintptr_t)dynlib.getAddressOfSymbol(
              "_ZN4lldb16PluginInitializeENS_10SBDebuggerE");
      if (init_func) {
        if (init_func(debugger_sb))
          return dynlib;
        else
          error = Status::FromErrorString(
              "plug-in refused to load "
              "(lldb::PluginInitialize(lldb::SBDebugger) "
              "returned false)");
      } else {
        error = Status::FromErrorString(
            "plug-in is missing the required initialization: "
            "lldb::PluginInitialize(lldb::SBDebugger)");
      }
    } else {
      if (FileSystem::Instance().Exists(spec))
        error = Status::FromErrorString(
            "this file does not represent a loadable dylib");
      else
        error = Status::FromErrorString("no such file");
    }
    return llvm::sys::DynamicLibrary();
  };

  Debugger::Initialize(LoadPlugin);

  return llvm::Error::success();
}

void SystemInitializerFull::Terminate() {
  Debugger::Terminate();

  Debugger::SettingsTerminate();

  // Terminate plug-ins in core LLDB.
  ProcessTrace::Terminate();

  // Terminate and unload and loaded system or user LLDB plug-ins.
  PluginManager::Terminate();

#define LLDB_PLUGIN(p) LLDB_PLUGIN_TERMINATE(p);
#include "Plugins/Plugins.def"

  // Now shutdown the common parts, in reverse order.
  SystemInitializerCommon::Terminate();
}
