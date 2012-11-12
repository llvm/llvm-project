//===-- PluginManager.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef liblldb_PluginManager_h_
#define liblldb_PluginManager_h_

#include "lldb/lldb-private.h"
#include "lldb/Host/FileSpec.h"

namespace lldb_private {

class PluginManager
{
public:
    static void
    Initialize ();
    
    static void
    Terminate ();

    //------------------------------------------------------------------
    // ABI
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    ABICreateInstance create_callback);

    static bool
    UnregisterPlugin (ABICreateInstance create_callback);

    static ABICreateInstance
    GetABICreateCallbackAtIndex (uint32_t idx);

    static ABICreateInstance
    GetABICreateCallbackForPluginName (const char *name);


    //------------------------------------------------------------------
    // Disassembler
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    DisassemblerCreateInstance create_callback);

    static bool
    UnregisterPlugin (DisassemblerCreateInstance create_callback);

    static DisassemblerCreateInstance
    GetDisassemblerCreateCallbackAtIndex (uint32_t idx);

    static DisassemblerCreateInstance
    GetDisassemblerCreateCallbackForPluginName (const char *name);


    //------------------------------------------------------------------
    // DynamicLoader
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    DynamicLoaderCreateInstance create_callback,
                    DebuggerInitializeCallback debugger_init_callback = NULL);

    static bool
    UnregisterPlugin (DynamicLoaderCreateInstance create_callback);

    static DynamicLoaderCreateInstance
    GetDynamicLoaderCreateCallbackAtIndex (uint32_t idx);

    static DynamicLoaderCreateInstance
    GetDynamicLoaderCreateCallbackForPluginName (const char *name);

    //------------------------------------------------------------------
    // EmulateInstruction
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    EmulateInstructionCreateInstance create_callback);
    
    static bool
    UnregisterPlugin (EmulateInstructionCreateInstance create_callback);
    
    static EmulateInstructionCreateInstance
    GetEmulateInstructionCreateCallbackAtIndex (uint32_t idx);
    
    static EmulateInstructionCreateInstance
    GetEmulateInstructionCreateCallbackForPluginName (const char *name);

    //------------------------------------------------------------------
    // OperatingSystem
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    OperatingSystemCreateInstance create_callback);
    
    static bool
    UnregisterPlugin (OperatingSystemCreateInstance create_callback);
    
    static OperatingSystemCreateInstance
    GetOperatingSystemCreateCallbackAtIndex (uint32_t idx);
    
    static OperatingSystemCreateInstance
    GetOperatingSystemCreateCallbackForPluginName (const char *name);

    //------------------------------------------------------------------
    // LanguageRuntime
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    LanguageRuntimeCreateInstance create_callback);

    static bool
    UnregisterPlugin (LanguageRuntimeCreateInstance create_callback);

    static LanguageRuntimeCreateInstance
    GetLanguageRuntimeCreateCallbackAtIndex (uint32_t idx);

    static LanguageRuntimeCreateInstance
    GetLanguageRuntimeCreateCallbackForPluginName (const char *name);


    //------------------------------------------------------------------
    // ObjectFile
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    ObjectFileCreateInstance create_callback,
                    ObjectFileCreateMemoryInstance create_memory_callback);

    static bool
    UnregisterPlugin (ObjectFileCreateInstance create_callback);

    static ObjectFileCreateInstance
    GetObjectFileCreateCallbackAtIndex (uint32_t idx);
    
    static ObjectFileCreateMemoryInstance
    GetObjectFileCreateMemoryCallbackAtIndex (uint32_t idx);

    static ObjectFileCreateInstance
    GetObjectFileCreateCallbackForPluginName (const char *name);

    static ObjectFileCreateMemoryInstance
    GetObjectFileCreateMemoryCallbackForPluginName (const char *name);


    //------------------------------------------------------------------
    // ObjectContainer
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    ObjectContainerCreateInstance create_callback);

    static bool
    UnregisterPlugin (ObjectContainerCreateInstance create_callback);

    static ObjectContainerCreateInstance
    GetObjectContainerCreateCallbackAtIndex (uint32_t idx);

    static ObjectContainerCreateInstance
    GetObjectContainerCreateCallbackForPluginName (const char *name);

    //------------------------------------------------------------------
    // LogChannel
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    LogChannelCreateInstance create_callback);

    static bool
    UnregisterPlugin (LogChannelCreateInstance create_callback);

    static LogChannelCreateInstance
    GetLogChannelCreateCallbackAtIndex (uint32_t idx);

    static LogChannelCreateInstance
    GetLogChannelCreateCallbackForPluginName (const char *name);

    static const char *
    GetLogChannelCreateNameAtIndex (uint32_t idx);

    //------------------------------------------------------------------
    // Platform
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    PlatformCreateInstance create_callback);

    static bool
    UnregisterPlugin (PlatformCreateInstance create_callback);

    static PlatformCreateInstance
    GetPlatformCreateCallbackAtIndex (uint32_t idx);

    static PlatformCreateInstance
    GetPlatformCreateCallbackForPluginName (const char *name);
    
    static const char *
    GetPlatformPluginNameAtIndex (uint32_t idx);

    static const char *
    GetPlatformPluginDescriptionAtIndex (uint32_t idx);

    static uint32_t
    AutoCompletePlatformName (const char *partial_name, 
                              StringList &matches);
    //------------------------------------------------------------------
    // Process
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    ProcessCreateInstance create_callback);
    
    static bool
    UnregisterPlugin (ProcessCreateInstance create_callback);
    
    static ProcessCreateInstance
    GetProcessCreateCallbackAtIndex (uint32_t idx);
    
    static ProcessCreateInstance
    GetProcessCreateCallbackForPluginName (const char *name);
    
    static const char *
    GetProcessPluginNameAtIndex (uint32_t idx);
    
    static const char *
    GetProcessPluginDescriptionAtIndex (uint32_t idx);

    //------------------------------------------------------------------
    // SymbolFile
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    SymbolFileCreateInstance create_callback);

    static bool
    UnregisterPlugin (SymbolFileCreateInstance create_callback);

    static SymbolFileCreateInstance
    GetSymbolFileCreateCallbackAtIndex (uint32_t idx);

    static SymbolFileCreateInstance
    GetSymbolFileCreateCallbackForPluginName (const char *name);


    //------------------------------------------------------------------
    // SymbolVendor
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    SymbolVendorCreateInstance create_callback);

    static bool
    UnregisterPlugin (SymbolVendorCreateInstance create_callback);

    static SymbolVendorCreateInstance
    GetSymbolVendorCreateCallbackAtIndex (uint32_t idx);

    static SymbolVendorCreateInstance
    GetSymbolVendorCreateCallbackForPluginName (const char *name);

    //------------------------------------------------------------------
    // UnwindAssembly
    //------------------------------------------------------------------
    static bool
    RegisterPlugin (const char *name,
                    const char *description,
                    UnwindAssemblyCreateInstance create_callback);

    static bool
    UnregisterPlugin (UnwindAssemblyCreateInstance create_callback);

    static UnwindAssemblyCreateInstance
    GetUnwindAssemblyCreateCallbackAtIndex (uint32_t idx);

    static UnwindAssemblyCreateInstance
    GetUnwindAssemblyCreateCallbackForPluginName (const char *name);

    //------------------------------------------------------------------
    // Some plug-ins might register a DebuggerInitializeCallback
    // callback when registering the plug-in. After a new Debugger
    // instance is created, this DebuggerInitialize function will get
    // called. This allows plug-ins to install Properties and do any
    // other intialization that requires a debugger instance.
    //------------------------------------------------------------------
    static void
    DebuggerInitialize (Debugger &debugger);
    
    static lldb::OptionValuePropertiesSP
    GetSettingForDynamicLoaderPlugin (Debugger &debugger,
                                       const ConstString &setting_name);
    
    static bool
    CreateSettingForDynamicLoaderPlugin (Debugger &debugger,
                                         const lldb::OptionValuePropertiesSP &properties_sp,
                                         const ConstString &description,
                                         bool is_global_property);
};


} // namespace lldb_private

#endif  // liblldb_PluginManager_h_
