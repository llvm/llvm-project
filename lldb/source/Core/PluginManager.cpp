//===-- PluginManager.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"

#include <limits.h>

#include <string>
#include <vector>

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Interpreter/OptionValueProperties.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

enum PluginAction
{
    ePluginRegisterInstance,
    ePluginUnregisterInstance,
    ePluginGetInstanceAtIndex
};

struct PluginInfo
{
    void *plugin_handle;
    void *plugin_init_callback;
    void *plugin_term_callback;
};

typedef std::map<FileSpec, PluginInfo> PluginTerminateMap;

static Mutex &
GetPluginMapMutex ()
{
    static Mutex g_plugin_map_mutex (Mutex::eMutexTypeRecursive);
    return g_plugin_map_mutex;
}

static PluginTerminateMap &
GetPluginMap ()
{
    static PluginTerminateMap g_plugin_map;
    return g_plugin_map;
}

static bool
PluginIsLoaded (const FileSpec &plugin_file_spec)
{
    Mutex::Locker locker (GetPluginMapMutex ());
    PluginTerminateMap &plugin_map = GetPluginMap ();
    return plugin_map.find (plugin_file_spec) != plugin_map.end();
}
    
static void
SetPluginInfo (const FileSpec &plugin_file_spec, const PluginInfo &plugin_info)
{
    Mutex::Locker locker (GetPluginMapMutex ());
    PluginTerminateMap &plugin_map = GetPluginMap ();
    assert (plugin_map.find (plugin_file_spec) != plugin_map.end());
    plugin_map[plugin_file_spec] = plugin_info;
}


static FileSpec::EnumerateDirectoryResult 
LoadPluginCallback 
(
    void *baton,
    FileSpec::FileType file_type,
    const FileSpec &file_spec
)
{
//    PluginManager *plugin_manager = (PluginManager *)baton;
    Error error;
    
    // If we have a regular file, a symbolic link or unknown file type, try
    // and process the file. We must handle unknown as sometimes the directory 
    // enumeration might be enumerating a file system that doesn't have correct
    // file type information.
    if (file_type == FileSpec::eFileTypeRegular         ||
        file_type == FileSpec::eFileTypeSymbolicLink    ||
        file_type == FileSpec::eFileTypeUnknown          )
    {
        FileSpec plugin_file_spec (file_spec);
        plugin_file_spec.ResolvePath();
        
        if (PluginIsLoaded (plugin_file_spec))
            return FileSpec::eEnumerateDirectoryResultNext;
        else
        {
            PluginInfo plugin_info = { NULL, NULL, NULL };
            uint32_t flags = Host::eDynamicLibraryOpenOptionLazy |
                             Host::eDynamicLibraryOpenOptionLocal |
                             Host::eDynamicLibraryOpenOptionLimitGetSymbol;

            plugin_info.plugin_handle = Host::DynamicLibraryOpen (plugin_file_spec, flags, error);
            if (plugin_info.plugin_handle)
            {
                bool success = false;
                plugin_info.plugin_init_callback = Host::DynamicLibraryGetSymbol (plugin_info.plugin_handle, "LLDBPluginInitialize", error);
                if (plugin_info.plugin_init_callback)
                {
                    // Call the plug-in "bool LLDBPluginInitialize(void)" function
                    success = ((bool (*)(void))plugin_info.plugin_init_callback)();
                }

                if (success)
                {
                    // It is ok for the "LLDBPluginTerminate" symbol to be NULL
                    plugin_info.plugin_term_callback = Host::DynamicLibraryGetSymbol (plugin_info.plugin_handle, "LLDBPluginTerminate", error);
                }
                else 
                {
                    // The initialize function returned FALSE which means the
                    // plug-in might not be compatible, or might be too new or
                    // too old, or might not want to run on this machine.
                    Host::DynamicLibraryClose (plugin_info.plugin_handle);
                    plugin_info.plugin_handle = NULL;
                    plugin_info.plugin_init_callback = NULL;
                }

                // Regardless of success or failure, cache the plug-in load
                // in our plug-in info so we don't try to load it again and 
                // again.
                SetPluginInfo (plugin_file_spec, plugin_info);

                return FileSpec::eEnumerateDirectoryResultNext;
            }
        }
    }
    
    if (file_type == FileSpec::eFileTypeUnknown     ||
        file_type == FileSpec::eFileTypeDirectory   ||
        file_type == FileSpec::eFileTypeSymbolicLink )
    {
        // Try and recurse into anything that a directory or symbolic link. 
        // We must also do this for unknown as sometimes the directory enumeration
        // might be enurating a file system that doesn't have correct file type
        // information.
        return FileSpec::eEnumerateDirectoryResultEnter;
    }

    return FileSpec::eEnumerateDirectoryResultNext;
}


void
PluginManager::Initialize ()
{
#if 1
    FileSpec dir_spec;
    const bool find_directories = true;
    const bool find_files = true;
    const bool find_other = true;
    char dir_path[PATH_MAX];
    if (Host::GetLLDBPath (ePathTypeLLDBSystemPlugins, dir_spec))
    {
        if (dir_spec.Exists() && dir_spec.GetPath(dir_path, sizeof(dir_path)))
        {
            FileSpec::EnumerateDirectory (dir_path, 
                                          find_directories,
                                          find_files,
                                          find_other,
                                          LoadPluginCallback,
                                          NULL);
        }
    }

    if (Host::GetLLDBPath (ePathTypeLLDBUserPlugins, dir_spec))
    {
        if (dir_spec.Exists() && dir_spec.GetPath(dir_path, sizeof(dir_path)))
        {
            FileSpec::EnumerateDirectory (dir_path, 
                                          find_directories,
                                          find_files,
                                          find_other,
                                          LoadPluginCallback,
                                          NULL);
        }
    }
#endif
}

void
PluginManager::Terminate ()
{
    Mutex::Locker locker (GetPluginMapMutex ());
    PluginTerminateMap &plugin_map = GetPluginMap ();
    
    PluginTerminateMap::const_iterator pos, end = plugin_map.end();
    for (pos = plugin_map.begin(); pos != end; ++pos)
    {
        // Call the plug-in "void LLDBPluginTerminate (void)" function if there
        // is one (if the symbol was not NULL).
        if (pos->second.plugin_handle)
        {
            if (pos->second.plugin_term_callback)
                ((void (*)(void))pos->second.plugin_term_callback)();
            Host::DynamicLibraryClose (pos->second.plugin_handle);
        }
    }
    plugin_map.clear();
}


#pragma mark ABI


struct ABIInstance
{
    ABIInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    ABICreateInstance create_callback;
};

typedef std::vector<ABIInstance> ABIInstances;

static Mutex &
GetABIInstancesMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static ABIInstances &
GetABIInstances ()
{
    static ABIInstances g_instances;
    return g_instances;
}

bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    ABICreateInstance create_callback
)
{
    if (create_callback)
    {
        ABIInstance instance;
        assert (name && name[0]);
        instance.name.assign (name);
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        Mutex::Locker locker (GetABIInstancesMutex ());
        GetABIInstances ().push_back (instance);
        return true;
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (ABICreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetABIInstancesMutex ());
        ABIInstances &instances = GetABIInstances ();

        ABIInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

ABICreateInstance
PluginManager::GetABICreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetABIInstancesMutex ());
    ABIInstances &instances = GetABIInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}

ABICreateInstance
PluginManager::GetABICreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        Mutex::Locker locker (GetABIInstancesMutex ());
        llvm::StringRef name_sref(name);
        ABIInstances &instances = GetABIInstances ();

        ABIInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}


#pragma mark Disassembler


struct DisassemblerInstance
{
    DisassemblerInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    DisassemblerCreateInstance create_callback;
};

typedef std::vector<DisassemblerInstance> DisassemblerInstances;

static Mutex &
GetDisassemblerMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static DisassemblerInstances &
GetDisassemblerInstances ()
{
    static DisassemblerInstances g_instances;
    return g_instances;
}

bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    DisassemblerCreateInstance create_callback
)
{
    if (create_callback)
    {
        DisassemblerInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        Mutex::Locker locker (GetDisassemblerMutex ());
        GetDisassemblerInstances ().push_back (instance);
        return true;
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (DisassemblerCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetDisassemblerMutex ());
        DisassemblerInstances &instances = GetDisassemblerInstances ();
        
        DisassemblerInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

DisassemblerCreateInstance
PluginManager::GetDisassemblerCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetDisassemblerMutex ());
    DisassemblerInstances &instances = GetDisassemblerInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}

DisassemblerCreateInstance
PluginManager::GetDisassemblerCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetDisassemblerMutex ());
        DisassemblerInstances &instances = GetDisassemblerInstances ();
        
        DisassemblerInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}



#pragma mark DynamicLoader


struct DynamicLoaderInstance
{
    DynamicLoaderInstance() :
        name(),
        description(),
        create_callback(NULL),
        debugger_init_callback (NULL)
    {
    }

    std::string name;
    std::string description;
    DynamicLoaderCreateInstance create_callback;
    DebuggerInitializeCallback debugger_init_callback;
};

typedef std::vector<DynamicLoaderInstance> DynamicLoaderInstances;


static Mutex &
GetDynamicLoaderMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static DynamicLoaderInstances &
GetDynamicLoaderInstances ()
{
    static DynamicLoaderInstances g_instances;
    return g_instances;
}


bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    DynamicLoaderCreateInstance create_callback,
    DebuggerInitializeCallback debugger_init_callback
)
{
    if (create_callback)
    {
        DynamicLoaderInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        instance.debugger_init_callback = debugger_init_callback;
        Mutex::Locker locker (GetDynamicLoaderMutex ());
        GetDynamicLoaderInstances ().push_back (instance);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (DynamicLoaderCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetDynamicLoaderMutex ());
        DynamicLoaderInstances &instances = GetDynamicLoaderInstances ();
        
        DynamicLoaderInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

DynamicLoaderCreateInstance
PluginManager::GetDynamicLoaderCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetDynamicLoaderMutex ());
    DynamicLoaderInstances &instances = GetDynamicLoaderInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}

DynamicLoaderCreateInstance
PluginManager::GetDynamicLoaderCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetDynamicLoaderMutex ());
        DynamicLoaderInstances &instances = GetDynamicLoaderInstances ();
        
        DynamicLoaderInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}

#pragma mark EmulateInstruction


struct EmulateInstructionInstance
{
    EmulateInstructionInstance() :
    name(),
    description(),
    create_callback(NULL)
    {
    }
    
    std::string name;
    std::string description;
    EmulateInstructionCreateInstance create_callback;
};

typedef std::vector<EmulateInstructionInstance> EmulateInstructionInstances;

static Mutex &
GetEmulateInstructionMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static EmulateInstructionInstances &
GetEmulateInstructionInstances ()
{
    static EmulateInstructionInstances g_instances;
    return g_instances;
}


bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    EmulateInstructionCreateInstance create_callback
)
{
    if (create_callback)
    {
        EmulateInstructionInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        Mutex::Locker locker (GetEmulateInstructionMutex ());
        GetEmulateInstructionInstances ().push_back (instance);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (EmulateInstructionCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetEmulateInstructionMutex ());
        EmulateInstructionInstances &instances = GetEmulateInstructionInstances ();
        
        EmulateInstructionInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

EmulateInstructionCreateInstance
PluginManager::GetEmulateInstructionCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetEmulateInstructionMutex ());
    EmulateInstructionInstances &instances = GetEmulateInstructionInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}

EmulateInstructionCreateInstance
PluginManager::GetEmulateInstructionCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetEmulateInstructionMutex ());
        EmulateInstructionInstances &instances = GetEmulateInstructionInstances ();
        
        EmulateInstructionInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}
#pragma mark OperatingSystem


struct OperatingSystemInstance
{
    OperatingSystemInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }
    
    std::string name;
    std::string description;
    OperatingSystemCreateInstance create_callback;
};

typedef std::vector<OperatingSystemInstance> OperatingSystemInstances;

static Mutex &
GetOperatingSystemMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static OperatingSystemInstances &
GetOperatingSystemInstances ()
{
    static OperatingSystemInstances g_instances;
    return g_instances;
}

bool
PluginManager::RegisterPlugin
(
 const char *name,
 const char *description,
 OperatingSystemCreateInstance create_callback
 )
{
    if (create_callback)
    {
        OperatingSystemInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        Mutex::Locker locker (GetOperatingSystemMutex ());
        GetOperatingSystemInstances ().push_back (instance);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (OperatingSystemCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetOperatingSystemMutex ());
        OperatingSystemInstances &instances = GetOperatingSystemInstances ();
        
        OperatingSystemInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

OperatingSystemCreateInstance
PluginManager::GetOperatingSystemCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetOperatingSystemMutex ());
    OperatingSystemInstances &instances = GetOperatingSystemInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}

OperatingSystemCreateInstance
PluginManager::GetOperatingSystemCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetOperatingSystemMutex ());
        OperatingSystemInstances &instances = GetOperatingSystemInstances ();
        
        OperatingSystemInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}


#pragma mark LanguageRuntime


struct LanguageRuntimeInstance
{
    LanguageRuntimeInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    LanguageRuntimeCreateInstance create_callback;
};

typedef std::vector<LanguageRuntimeInstance> LanguageRuntimeInstances;

static Mutex &
GetLanguageRuntimeMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static LanguageRuntimeInstances &
GetLanguageRuntimeInstances ()
{
    static LanguageRuntimeInstances g_instances;
    return g_instances;
}

bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    LanguageRuntimeCreateInstance create_callback
)
{
    if (create_callback)
    {
        LanguageRuntimeInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        Mutex::Locker locker (GetLanguageRuntimeMutex ());
        GetLanguageRuntimeInstances ().push_back (instance);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (LanguageRuntimeCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetLanguageRuntimeMutex ());
        LanguageRuntimeInstances &instances = GetLanguageRuntimeInstances ();
        
        LanguageRuntimeInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

LanguageRuntimeCreateInstance
PluginManager::GetLanguageRuntimeCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetLanguageRuntimeMutex ());
    LanguageRuntimeInstances &instances = GetLanguageRuntimeInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}

LanguageRuntimeCreateInstance
PluginManager::GetLanguageRuntimeCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetLanguageRuntimeMutex ());
        LanguageRuntimeInstances &instances = GetLanguageRuntimeInstances ();
        
        LanguageRuntimeInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}

#pragma mark ObjectFile

struct ObjectFileInstance
{
    ObjectFileInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    ObjectFileCreateInstance create_callback;
    ObjectFileCreateMemoryInstance create_memory_callback;

};

typedef std::vector<ObjectFileInstance> ObjectFileInstances;

static Mutex &
GetObjectFileMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static ObjectFileInstances &
GetObjectFileInstances ()
{
    static ObjectFileInstances g_instances;
    return g_instances;
}


bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    ObjectFileCreateInstance create_callback,
    ObjectFileCreateMemoryInstance create_memory_callback
)
{
    if (create_callback)
    {
        ObjectFileInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        instance.create_memory_callback = create_memory_callback;
        Mutex::Locker locker (GetObjectFileMutex ());
        GetObjectFileInstances ().push_back (instance);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (ObjectFileCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetObjectFileMutex ());
        ObjectFileInstances &instances = GetObjectFileInstances ();
        
        ObjectFileInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

ObjectFileCreateInstance
PluginManager::GetObjectFileCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetObjectFileMutex ());
    ObjectFileInstances &instances = GetObjectFileInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}


ObjectFileCreateMemoryInstance
PluginManager::GetObjectFileCreateMemoryCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetObjectFileMutex ());
    ObjectFileInstances &instances = GetObjectFileInstances ();
    if (idx < instances.size())
        return instances[idx].create_memory_callback;
    return NULL;
}

ObjectFileCreateInstance
PluginManager::GetObjectFileCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetObjectFileMutex ());
        ObjectFileInstances &instances = GetObjectFileInstances ();
        
        ObjectFileInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}


ObjectFileCreateMemoryInstance
PluginManager::GetObjectFileCreateMemoryCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetObjectFileMutex ());
        ObjectFileInstances &instances = GetObjectFileInstances ();
        
        ObjectFileInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_memory_callback;
        }
    }
    return NULL;
}



#pragma mark ObjectContainer

struct ObjectContainerInstance
{
    ObjectContainerInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    ObjectContainerCreateInstance create_callback;
};

typedef std::vector<ObjectContainerInstance> ObjectContainerInstances;

static Mutex &
GetObjectContainerMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static ObjectContainerInstances &
GetObjectContainerInstances ()
{
    static ObjectContainerInstances g_instances;
    return g_instances;
}

bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    ObjectContainerCreateInstance create_callback
)
{
    if (create_callback)
    {
        ObjectContainerInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        Mutex::Locker locker (GetObjectContainerMutex ());
        GetObjectContainerInstances ().push_back (instance);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (ObjectContainerCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetObjectContainerMutex ());
        ObjectContainerInstances &instances = GetObjectContainerInstances ();
        
        ObjectContainerInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

ObjectContainerCreateInstance
PluginManager::GetObjectContainerCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetObjectContainerMutex ());
    ObjectContainerInstances &instances = GetObjectContainerInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}

ObjectContainerCreateInstance
PluginManager::GetObjectContainerCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetObjectContainerMutex ());
        ObjectContainerInstances &instances = GetObjectContainerInstances ();
        
        ObjectContainerInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}

#pragma mark LogChannel

struct LogInstance
{
    LogInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    LogChannelCreateInstance create_callback;
};

typedef std::vector<LogInstance> LogInstances;

static Mutex &
GetLogMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static LogInstances &
GetLogInstances ()
{
    static LogInstances g_instances;
    return g_instances;
}



bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    LogChannelCreateInstance create_callback
)
{
    if (create_callback)
    {
        LogInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        Mutex::Locker locker (GetLogMutex ());
        GetLogInstances ().push_back (instance);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (LogChannelCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetLogMutex ());
        LogInstances &instances = GetLogInstances ();
        
        LogInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

const char *
PluginManager::GetLogChannelCreateNameAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetLogMutex ());
    LogInstances &instances = GetLogInstances ();
    if (idx < instances.size())
        return instances[idx].name.c_str();
    return NULL;
}


LogChannelCreateInstance
PluginManager::GetLogChannelCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetLogMutex ());
    LogInstances &instances = GetLogInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}

LogChannelCreateInstance
PluginManager::GetLogChannelCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetLogMutex ());
        LogInstances &instances = GetLogInstances ();
        
        LogInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}

#pragma mark Platform

struct PlatformInstance
{
    PlatformInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }
    
    std::string name;
    std::string description;
    PlatformCreateInstance create_callback;
};

typedef std::vector<PlatformInstance> PlatformInstances;

static Mutex &
GetPlatformInstancesMutex ()
{
    static Mutex g_platform_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_platform_instances_mutex;
}

static PlatformInstances &
GetPlatformInstances ()
{
    static PlatformInstances g_platform_instances;
    return g_platform_instances;
}


bool
PluginManager::RegisterPlugin (const char *name,
                               const char *description,
                               PlatformCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetPlatformInstancesMutex ());
        
        PlatformInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        GetPlatformInstances ().push_back (instance);
        return true;
    }
    return false;
}

const char *
PluginManager::GetPlatformPluginNameAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetPlatformInstancesMutex ());
    PlatformInstances &instances = GetPlatformInstances ();
    if (idx < instances.size())
        return instances[idx].name.c_str();
    return NULL;
}

const char *
PluginManager::GetPlatformPluginDescriptionAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetPlatformInstancesMutex ());
    PlatformInstances &instances = GetPlatformInstances ();
    if (idx < instances.size())
        return instances[idx].description.c_str();
    return NULL;
}

bool
PluginManager::UnregisterPlugin (PlatformCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetPlatformInstancesMutex ());
        PlatformInstances &instances = GetPlatformInstances ();

        PlatformInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

PlatformCreateInstance
PluginManager::GetPlatformCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetPlatformInstancesMutex ());
    PlatformInstances &instances = GetPlatformInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}

PlatformCreateInstance
PluginManager::GetPlatformCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        Mutex::Locker locker (GetPlatformInstancesMutex ());
        PlatformInstances &instances = GetPlatformInstances ();
        llvm::StringRef name_sref(name);

        PlatformInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}

uint32_t
PluginManager::AutoCompletePlatformName (const char *name, StringList &matches)
{
    if (name && name[0])
    {
        Mutex::Locker locker (GetPlatformInstancesMutex ());
        PlatformInstances &instances = GetPlatformInstances ();
        llvm::StringRef name_sref(name);

        PlatformInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            llvm::StringRef plugin_name (pos->name);
            if (plugin_name.startswith(name_sref))
                matches.AppendString (plugin_name.data());
        }
    }
    return matches.GetSize();
}

#pragma mark Process

struct ProcessInstance
{
    ProcessInstance() :
    name(),
    description(),
    create_callback(NULL)
    {
    }
    
    std::string name;
    std::string description;
    ProcessCreateInstance create_callback;
};

typedef std::vector<ProcessInstance> ProcessInstances;

static Mutex &
GetProcessMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static ProcessInstances &
GetProcessInstances ()
{
    static ProcessInstances g_instances;
    return g_instances;
}


bool
PluginManager::RegisterPlugin
(
 const char *name,
 const char *description,
 ProcessCreateInstance create_callback
 )
{
    if (create_callback)
    {
        ProcessInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        Mutex::Locker locker (GetProcessMutex ());
        GetProcessInstances ().push_back (instance);
    }
    return false;
}

const char *
PluginManager::GetProcessPluginNameAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetProcessMutex ());
    ProcessInstances &instances = GetProcessInstances ();
    if (idx < instances.size())
        return instances[idx].name.c_str();
    return NULL;
}

const char *
PluginManager::GetProcessPluginDescriptionAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetProcessMutex ());
    ProcessInstances &instances = GetProcessInstances ();
    if (idx < instances.size())
        return instances[idx].description.c_str();
    return NULL;
}

bool
PluginManager::UnregisterPlugin (ProcessCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetProcessMutex ());
        ProcessInstances &instances = GetProcessInstances ();
        
        ProcessInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

ProcessCreateInstance
PluginManager::GetProcessCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetProcessMutex ());
    ProcessInstances &instances = GetProcessInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}


ProcessCreateInstance
PluginManager::GetProcessCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetProcessMutex ());
        ProcessInstances &instances = GetProcessInstances ();
        
        ProcessInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}

#pragma mark SymbolFile

struct SymbolFileInstance
{
    SymbolFileInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    SymbolFileCreateInstance create_callback;
};

typedef std::vector<SymbolFileInstance> SymbolFileInstances;

static Mutex &
GetSymbolFileMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static SymbolFileInstances &
GetSymbolFileInstances ()
{
    static SymbolFileInstances g_instances;
    return g_instances;
}


bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    SymbolFileCreateInstance create_callback
)
{
    if (create_callback)
    {
        SymbolFileInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        Mutex::Locker locker (GetSymbolFileMutex ());
        GetSymbolFileInstances ().push_back (instance);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (SymbolFileCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetSymbolFileMutex ());
        SymbolFileInstances &instances = GetSymbolFileInstances ();
        
        SymbolFileInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

SymbolFileCreateInstance
PluginManager::GetSymbolFileCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetSymbolFileMutex ());
    SymbolFileInstances &instances = GetSymbolFileInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}

SymbolFileCreateInstance
PluginManager::GetSymbolFileCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetSymbolFileMutex ());
        SymbolFileInstances &instances = GetSymbolFileInstances ();
        
        SymbolFileInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}



#pragma mark SymbolVendor

struct SymbolVendorInstance
{
    SymbolVendorInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    SymbolVendorCreateInstance create_callback;
};

typedef std::vector<SymbolVendorInstance> SymbolVendorInstances;

static Mutex &
GetSymbolVendorMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static SymbolVendorInstances &
GetSymbolVendorInstances ()
{
    static SymbolVendorInstances g_instances;
    return g_instances;
}

bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    SymbolVendorCreateInstance create_callback
)
{
    if (create_callback)
    {
        SymbolVendorInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        Mutex::Locker locker (GetSymbolVendorMutex ());
        GetSymbolVendorInstances ().push_back (instance);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (SymbolVendorCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetSymbolVendorMutex ());
        SymbolVendorInstances &instances = GetSymbolVendorInstances ();
        
        SymbolVendorInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

SymbolVendorCreateInstance
PluginManager::GetSymbolVendorCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetSymbolVendorMutex ());
    SymbolVendorInstances &instances = GetSymbolVendorInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}


SymbolVendorCreateInstance
PluginManager::GetSymbolVendorCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetSymbolVendorMutex ());
        SymbolVendorInstances &instances = GetSymbolVendorInstances ();
        
        SymbolVendorInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}


#pragma mark UnwindAssembly

struct UnwindAssemblyInstance
{
    UnwindAssemblyInstance() :
        name(),
        description(),
        create_callback(NULL)
    {
    }

    std::string name;
    std::string description;
    UnwindAssemblyCreateInstance create_callback;
};

typedef std::vector<UnwindAssemblyInstance> UnwindAssemblyInstances;

static Mutex &
GetUnwindAssemblyMutex ()
{
    static Mutex g_instances_mutex (Mutex::eMutexTypeRecursive);
    return g_instances_mutex;
}

static UnwindAssemblyInstances &
GetUnwindAssemblyInstances ()
{
    static UnwindAssemblyInstances g_instances;
    return g_instances;
}

bool
PluginManager::RegisterPlugin
(
    const char *name,
    const char *description,
    UnwindAssemblyCreateInstance create_callback
)
{
    if (create_callback)
    {
        UnwindAssemblyInstance instance;
        assert (name && name[0]);
        instance.name = name;
        if (description && description[0])
            instance.description = description;
        instance.create_callback = create_callback;
        Mutex::Locker locker (GetUnwindAssemblyMutex ());
        GetUnwindAssemblyInstances ().push_back (instance);
    }
    return false;
}

bool
PluginManager::UnregisterPlugin (UnwindAssemblyCreateInstance create_callback)
{
    if (create_callback)
    {
        Mutex::Locker locker (GetUnwindAssemblyMutex ());
        UnwindAssemblyInstances &instances = GetUnwindAssemblyInstances ();
        
        UnwindAssemblyInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (pos->create_callback == create_callback)
            {
                instances.erase(pos);
                return true;
            }
        }
    }
    return false;
}

UnwindAssemblyCreateInstance
PluginManager::GetUnwindAssemblyCreateCallbackAtIndex (uint32_t idx)
{
    Mutex::Locker locker (GetUnwindAssemblyMutex ());
    UnwindAssemblyInstances &instances = GetUnwindAssemblyInstances ();
    if (idx < instances.size())
        return instances[idx].create_callback;
    return NULL;
}


UnwindAssemblyCreateInstance
PluginManager::GetUnwindAssemblyCreateCallbackForPluginName (const char *name)
{
    if (name && name[0])
    {
        llvm::StringRef name_sref(name);
        Mutex::Locker locker (GetUnwindAssemblyMutex ());
        UnwindAssemblyInstances &instances = GetUnwindAssemblyInstances ();
        
        UnwindAssemblyInstances::iterator pos, end = instances.end();
        for (pos = instances.begin(); pos != end; ++ pos)
        {
            if (name_sref.equals (pos->name))
                return pos->create_callback;
        }
    }
    return NULL;
}

void
PluginManager::DebuggerInitialize (Debugger &debugger)
{
    Mutex::Locker locker (GetDynamicLoaderMutex ());
    DynamicLoaderInstances &instances = GetDynamicLoaderInstances ();
    
    DynamicLoaderInstances::iterator pos, end = instances.end();
    for (pos = instances.begin(); pos != end; ++ pos)
    {
        if (pos->debugger_init_callback)
            pos->debugger_init_callback (debugger);
    }
}

static lldb::OptionValuePropertiesSP
GetDebuggerPropertyForPlugins (Debugger &debugger,
                               const ConstString &plugin_type_name,
                               const ConstString &plugin_type_desc,
                               bool can_create)
{
    lldb::OptionValuePropertiesSP parent_properties_sp (debugger.GetValueProperties());
    if (parent_properties_sp)
    {
        static ConstString g_property_name("plugin");
        
        OptionValuePropertiesSP plugin_properties_sp = parent_properties_sp->GetSubProperty (NULL, g_property_name);
        if (!plugin_properties_sp && can_create)
        {
            plugin_properties_sp.reset (new OptionValueProperties (g_property_name));
            parent_properties_sp->AppendProperty (g_property_name,
                                                  ConstString("Settings specify to plugins."),
                                                  true,
                                                  plugin_properties_sp);
        }
        
        if (plugin_properties_sp)
        {
            lldb::OptionValuePropertiesSP plugin_type_properties_sp = plugin_properties_sp->GetSubProperty (NULL, plugin_type_name);
            if (!plugin_type_properties_sp && can_create)
            {
                plugin_type_properties_sp.reset (new OptionValueProperties (plugin_type_name));
                plugin_properties_sp->AppendProperty (plugin_type_name,
                                                      plugin_type_desc,
                                                      true,
                                                      plugin_type_properties_sp);
            }
            return plugin_type_properties_sp;
        }
    }
    return lldb::OptionValuePropertiesSP();
}

lldb::OptionValuePropertiesSP
PluginManager::GetSettingForDynamicLoaderPlugin (Debugger &debugger, const ConstString &setting_name)
{
    lldb::OptionValuePropertiesSP properties_sp;
    lldb::OptionValuePropertiesSP plugin_type_properties_sp (GetDebuggerPropertyForPlugins (debugger,
                                                                                            ConstString("dynamic-loader"),
                                                                                            ConstString(), // not creating to so we don't need the description
                                                                                            false));
    if (plugin_type_properties_sp)
        properties_sp = plugin_type_properties_sp->GetSubProperty (NULL, setting_name);
    return properties_sp;
}

bool
PluginManager::CreateSettingForDynamicLoaderPlugin (Debugger &debugger,
                                                    const lldb::OptionValuePropertiesSP &properties_sp,
                                                    const ConstString &description,
                                                    bool is_global_property)
{
    if (properties_sp)
    {
        lldb::OptionValuePropertiesSP plugin_type_properties_sp (GetDebuggerPropertyForPlugins (debugger,
                                                                                                ConstString("dynamic-loader"),
                                                                                                ConstString("Settings for dynamic loader plug-ins"),
                                                                                                true));
        if (plugin_type_properties_sp)
        {
            plugin_type_properties_sp->AppendProperty (properties_sp->GetName(),
                                                       description,
                                                       is_global_property,
                                                       properties_sp);
            return true;
        }
    }
    return false;
}

