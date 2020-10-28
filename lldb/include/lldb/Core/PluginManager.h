//===-- PluginManager.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_PLUGINMANAGER_H
#define LLDB_CORE_PLUGINMANAGER_H

#include "lldb/Core/Architecture.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Utility/CompletionRequest.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private-interfaces.h"
#include "llvm/ADT/StringRef.h"

#include <stddef.h>
#include <stdint.h>

#define LLDB_PLUGIN_DEFINE_ADV(ClassName, PluginName)                          \
  namespace lldb_private {                                                     \
  void lldb_initialize_##PluginName() { ClassName::Initialize(); }             \
  void lldb_terminate_##PluginName() { ClassName::Terminate(); }               \
  }

#define LLDB_PLUGIN_DEFINE(PluginName)                                         \
  LLDB_PLUGIN_DEFINE_ADV(PluginName, PluginName)

// FIXME: Generate me with CMake
#define LLDB_PLUGIN_DECLARE(PluginName)                                        \
  namespace lldb_private {                                                     \
  extern void lldb_initialize_##PluginName();                                  \
  extern void lldb_terminate_##PluginName();                                   \
  }

#define LLDB_PLUGIN_INITIALIZE(PluginName) lldb_initialize_##PluginName()
#define LLDB_PLUGIN_TERMINATE(PluginName) lldb_terminate_##PluginName()

namespace lldb_private {
class CommandInterpreter;
class ConstString;
class Debugger;
class StringList;

class PluginManager {
public:
  static void Initialize();

  static void Terminate();

  // ABI
  static bool RegisterPlugin(ConstString name, const char *description,
                             ABICreateInstance create_callback);

  static bool UnregisterPlugin(ABICreateInstance create_callback);

  static ABICreateInstance GetABICreateCallbackAtIndex(uint32_t idx);

  // Architecture
  static void RegisterPlugin(ConstString name, llvm::StringRef description,
                             ArchitectureCreateInstance create_callback);

  static void UnregisterPlugin(ArchitectureCreateInstance create_callback);

  static std::unique_ptr<Architecture>
  CreateArchitectureInstance(const ArchSpec &arch);

  // Disassembler
  static bool RegisterPlugin(ConstString name, const char *description,
                             DisassemblerCreateInstance create_callback);

  static bool UnregisterPlugin(DisassemblerCreateInstance create_callback);

  static DisassemblerCreateInstance
  GetDisassemblerCreateCallbackAtIndex(uint32_t idx);

  static DisassemblerCreateInstance
  GetDisassemblerCreateCallbackForPluginName(ConstString name);

  // DynamicLoader
  static bool
  RegisterPlugin(ConstString name, const char *description,
                 DynamicLoaderCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(DynamicLoaderCreateInstance create_callback);

  static DynamicLoaderCreateInstance
  GetDynamicLoaderCreateCallbackAtIndex(uint32_t idx);

  static DynamicLoaderCreateInstance
  GetDynamicLoaderCreateCallbackForPluginName(ConstString name);

  // JITLoader
  static bool
  RegisterPlugin(ConstString name, const char *description,
                 JITLoaderCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(JITLoaderCreateInstance create_callback);

  static JITLoaderCreateInstance
  GetJITLoaderCreateCallbackAtIndex(uint32_t idx);

  // EmulateInstruction
  static bool RegisterPlugin(ConstString name, const char *description,
                             EmulateInstructionCreateInstance create_callback);

  static bool
  UnregisterPlugin(EmulateInstructionCreateInstance create_callback);

  static EmulateInstructionCreateInstance
  GetEmulateInstructionCreateCallbackAtIndex(uint32_t idx);

  static EmulateInstructionCreateInstance
  GetEmulateInstructionCreateCallbackForPluginName(ConstString name);

  // OperatingSystem
  static bool RegisterPlugin(ConstString name, const char *description,
                             OperatingSystemCreateInstance create_callback,
                             DebuggerInitializeCallback debugger_init_callback);

  static bool UnregisterPlugin(OperatingSystemCreateInstance create_callback);

  static OperatingSystemCreateInstance
  GetOperatingSystemCreateCallbackAtIndex(uint32_t idx);

  static OperatingSystemCreateInstance
  GetOperatingSystemCreateCallbackForPluginName(ConstString name);

  // Language
  static bool RegisterPlugin(ConstString name, const char *description,
                             LanguageCreateInstance create_callback);

  static bool UnregisterPlugin(LanguageCreateInstance create_callback);

  static LanguageCreateInstance GetLanguageCreateCallbackAtIndex(uint32_t idx);

  // LanguageRuntime
  static bool RegisterPlugin(
      ConstString name, const char *description,
      LanguageRuntimeCreateInstance create_callback,
      LanguageRuntimeGetCommandObject command_callback = nullptr,
      LanguageRuntimeGetExceptionPrecondition precondition_callback = nullptr);

  static bool UnregisterPlugin(LanguageRuntimeCreateInstance create_callback);

  static LanguageRuntimeCreateInstance
  GetLanguageRuntimeCreateCallbackAtIndex(uint32_t idx);

  static LanguageRuntimeGetCommandObject
  GetLanguageRuntimeGetCommandObjectAtIndex(uint32_t idx);

  static LanguageRuntimeGetExceptionPrecondition
  GetLanguageRuntimeGetExceptionPreconditionAtIndex(uint32_t idx);

  // SystemRuntime
  static bool RegisterPlugin(ConstString name, const char *description,
                             SystemRuntimeCreateInstance create_callback);

  static bool UnregisterPlugin(SystemRuntimeCreateInstance create_callback);

  static SystemRuntimeCreateInstance
  GetSystemRuntimeCreateCallbackAtIndex(uint32_t idx);

  // ObjectFile
  static bool
  RegisterPlugin(ConstString name, const char *description,
                 ObjectFileCreateInstance create_callback,
                 ObjectFileCreateMemoryInstance create_memory_callback,
                 ObjectFileGetModuleSpecifications get_module_specifications,
                 ObjectFileSaveCore save_core = nullptr);

  static bool UnregisterPlugin(ObjectFileCreateInstance create_callback);

  static ObjectFileCreateInstance
  GetObjectFileCreateCallbackAtIndex(uint32_t idx);

  static ObjectFileCreateMemoryInstance
  GetObjectFileCreateMemoryCallbackAtIndex(uint32_t idx);

  static ObjectFileGetModuleSpecifications
  GetObjectFileGetModuleSpecificationsCallbackAtIndex(uint32_t idx);

  static ObjectFileCreateMemoryInstance
  GetObjectFileCreateMemoryCallbackForPluginName(ConstString name);

  static Status SaveCore(const lldb::ProcessSP &process_sp,
                         const FileSpec &outfile);

  // ObjectContainer
  static bool
  RegisterPlugin(ConstString name, const char *description,
                 ObjectContainerCreateInstance create_callback,
                 ObjectFileGetModuleSpecifications get_module_specifications);

  static bool UnregisterPlugin(ObjectContainerCreateInstance create_callback);

  static ObjectContainerCreateInstance
  GetObjectContainerCreateCallbackAtIndex(uint32_t idx);

  static ObjectFileGetModuleSpecifications
  GetObjectContainerGetModuleSpecificationsCallbackAtIndex(uint32_t idx);

  // Platform
  static bool
  RegisterPlugin(ConstString name, const char *description,
                 PlatformCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(PlatformCreateInstance create_callback);

  static PlatformCreateInstance GetPlatformCreateCallbackAtIndex(uint32_t idx);

  static PlatformCreateInstance
  GetPlatformCreateCallbackForPluginName(ConstString name);

  static const char *GetPlatformPluginNameAtIndex(uint32_t idx);

  static const char *GetPlatformPluginDescriptionAtIndex(uint32_t idx);

  static void AutoCompletePlatformName(llvm::StringRef partial_name,
                                       CompletionRequest &request);
  // Process
  static bool
  RegisterPlugin(ConstString name, const char *description,
                 ProcessCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(ProcessCreateInstance create_callback);

  static ProcessCreateInstance GetProcessCreateCallbackAtIndex(uint32_t idx);

  static ProcessCreateInstance
  GetProcessCreateCallbackForPluginName(ConstString name);

  static const char *GetProcessPluginNameAtIndex(uint32_t idx);

  static const char *GetProcessPluginDescriptionAtIndex(uint32_t idx);

  static void AutoCompleteProcessName(llvm::StringRef partial_name,
                                      CompletionRequest &request);

  // ScriptInterpreter
  static bool RegisterPlugin(ConstString name, const char *description,
                             lldb::ScriptLanguage script_lang,
                             ScriptInterpreterCreateInstance create_callback);

  static bool UnregisterPlugin(ScriptInterpreterCreateInstance create_callback);

  static ScriptInterpreterCreateInstance
  GetScriptInterpreterCreateCallbackAtIndex(uint32_t idx);

  static lldb::ScriptInterpreterSP
  GetScriptInterpreterForLanguage(lldb::ScriptLanguage script_lang,
                                  Debugger &debugger);

  // StructuredDataPlugin

  /// Register a StructuredDataPlugin class along with optional
  /// callbacks for debugger initialization and Process launch info
  /// filtering and manipulation.
  ///
  /// \param[in] name
  ///    The name of the plugin.
  ///
  /// \param[in] description
  ///    A description string for the plugin.
  ///
  /// \param[in] create_callback
  ///    The callback that will be invoked to create an instance of
  ///    the callback.  This may not be nullptr.
  ///
  /// \param[in] debugger_init_callback
  ///    An optional callback that will be made when a Debugger
  ///    instance is initialized.
  ///
  /// \param[in] filter_callback
  ///    An optional callback that will be invoked before LLDB
  ///    launches a process for debugging.  The callback must
  ///    do the following:
  ///    1. Only do something if the plugin's behavior is enabled.
  ///    2. Only make changes for processes that are relevant to the
  ///       plugin.  The callback gets a pointer to the Target, which
  ///       can be inspected as needed.  The ProcessLaunchInfo is
  ///       provided in read-write mode, and may be modified by the
  ///       plugin if, for instance, additional environment variables
  ///       are needed to support the feature when enabled.
  ///
  /// \return
  ///    Returns true upon success; otherwise, false.
  static bool
  RegisterPlugin(ConstString name, const char *description,
                 StructuredDataPluginCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr,
                 StructuredDataFilterLaunchInfo filter_callback = nullptr);

  static bool
  UnregisterPlugin(StructuredDataPluginCreateInstance create_callback);

  static StructuredDataPluginCreateInstance
  GetStructuredDataPluginCreateCallbackAtIndex(uint32_t idx);

  static StructuredDataFilterLaunchInfo
  GetStructuredDataFilterCallbackAtIndex(uint32_t idx,
                                         bool &iteration_complete);

  // SymbolFile
  static bool
  RegisterPlugin(ConstString name, const char *description,
                 SymbolFileCreateInstance create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr);

  static bool UnregisterPlugin(SymbolFileCreateInstance create_callback);

  static SymbolFileCreateInstance
  GetSymbolFileCreateCallbackAtIndex(uint32_t idx);

  // SymbolVendor
  static bool RegisterPlugin(ConstString name, const char *description,
                             SymbolVendorCreateInstance create_callback);

  static bool UnregisterPlugin(SymbolVendorCreateInstance create_callback);

  static SymbolVendorCreateInstance
  GetSymbolVendorCreateCallbackAtIndex(uint32_t idx);

  // Trace
  static bool RegisterPlugin(ConstString name, const char *description,
                             TraceCreateInstance create_callback);

  static bool UnregisterPlugin(TraceCreateInstance create_callback);

  static TraceCreateInstance GetTraceCreateCallback(ConstString plugin_name);

  // UnwindAssembly
  static bool RegisterPlugin(ConstString name, const char *description,
                             UnwindAssemblyCreateInstance create_callback);

  static bool UnregisterPlugin(UnwindAssemblyCreateInstance create_callback);

  static UnwindAssemblyCreateInstance
  GetUnwindAssemblyCreateCallbackAtIndex(uint32_t idx);

  // MemoryHistory
  static bool RegisterPlugin(ConstString name, const char *description,
                             MemoryHistoryCreateInstance create_callback);

  static bool UnregisterPlugin(MemoryHistoryCreateInstance create_callback);

  static MemoryHistoryCreateInstance
  GetMemoryHistoryCreateCallbackAtIndex(uint32_t idx);

  // InstrumentationRuntime
  static bool
  RegisterPlugin(ConstString name, const char *description,
                 InstrumentationRuntimeCreateInstance create_callback,
                 InstrumentationRuntimeGetType get_type_callback);

  static bool
  UnregisterPlugin(InstrumentationRuntimeCreateInstance create_callback);

  static InstrumentationRuntimeGetType
  GetInstrumentationRuntimeGetTypeCallbackAtIndex(uint32_t idx);

  static InstrumentationRuntimeCreateInstance
  GetInstrumentationRuntimeCreateCallbackAtIndex(uint32_t idx);

  // TypeSystem
  static bool RegisterPlugin(ConstString name, const char *description,
                             TypeSystemCreateInstance create_callback,
                             LanguageSet supported_languages_for_types,
                             LanguageSet supported_languages_for_expressions);

  static bool UnregisterPlugin(TypeSystemCreateInstance create_callback);

  static TypeSystemCreateInstance
  GetTypeSystemCreateCallbackAtIndex(uint32_t idx);

  static LanguageSet GetAllTypeSystemSupportedLanguagesForTypes();

  static LanguageSet GetAllTypeSystemSupportedLanguagesForExpressions();

  // REPL
  static bool RegisterPlugin(ConstString name, const char *description,
                             REPLCreateInstance create_callback,
                             LanguageSet supported_languages);

  static bool UnregisterPlugin(REPLCreateInstance create_callback);

  static REPLCreateInstance GetREPLCreateCallbackAtIndex(uint32_t idx);

  static LanguageSet GetREPLAllTypeSystemSupportedLanguages();

  // Some plug-ins might register a DebuggerInitializeCallback callback when
  // registering the plug-in. After a new Debugger instance is created, this
  // DebuggerInitialize function will get called. This allows plug-ins to
  // install Properties and do any other initialization that requires a
  // debugger instance.
  static void DebuggerInitialize(Debugger &debugger);

  static lldb::OptionValuePropertiesSP
  GetSettingForDynamicLoaderPlugin(Debugger &debugger,
                                   ConstString setting_name);

  static bool CreateSettingForDynamicLoaderPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      ConstString description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForPlatformPlugin(Debugger &debugger, ConstString setting_name);

  static bool CreateSettingForPlatformPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      ConstString description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForProcessPlugin(Debugger &debugger, ConstString setting_name);

  static bool CreateSettingForProcessPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      ConstString description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForSymbolFilePlugin(Debugger &debugger, ConstString setting_name);

  static bool CreateSettingForSymbolFilePlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      ConstString description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForJITLoaderPlugin(Debugger &debugger, ConstString setting_name);

  static bool CreateSettingForJITLoaderPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      ConstString description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForOperatingSystemPlugin(Debugger &debugger,
                                     ConstString setting_name);

  static bool CreateSettingForOperatingSystemPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      ConstString description, bool is_global_property);

  static lldb::OptionValuePropertiesSP
  GetSettingForStructuredDataPlugin(Debugger &debugger,
                                    ConstString setting_name);

  static bool CreateSettingForStructuredDataPlugin(
      Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
      ConstString description, bool is_global_property);
};

} // namespace lldb_private

#endif // LLDB_CORE_PLUGINMANAGER_H
