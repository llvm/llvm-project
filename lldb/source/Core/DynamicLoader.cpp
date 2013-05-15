//===-- DynamicLoader.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Process.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;

DynamicLoader*
DynamicLoader::FindPlugin (Process *process, const char *plugin_name)
{
    DynamicLoaderCreateInstance create_callback = NULL;
    if (plugin_name)
    {
        ConstString const_plugin_name(plugin_name);
        create_callback  = PluginManager::GetDynamicLoaderCreateCallbackForPluginName (const_plugin_name);
        if (create_callback)
        {
            std::unique_ptr<DynamicLoader> instance_ap(create_callback(process, true));
            if (instance_ap.get())
                return instance_ap.release();
        }
    }
    else
    {
        for (uint32_t idx = 0; (create_callback = PluginManager::GetDynamicLoaderCreateCallbackAtIndex(idx)) != NULL; ++idx)
        {
            std::unique_ptr<DynamicLoader> instance_ap(create_callback(process, false));
            if (instance_ap.get())
                return instance_ap.release();
        }
    }
    return NULL;
}


//----------------------------------------------------------------------
// DynamicLoader constructor
//----------------------------------------------------------------------
DynamicLoader::DynamicLoader(Process *process) :
    m_process (process)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
DynamicLoader::~DynamicLoader()
{
}

//----------------------------------------------------------------------
// Accessosors to the global setting as to whether to stop at image
// (shared library) loading/unloading.
//----------------------------------------------------------------------
bool
DynamicLoader::GetStopWhenImagesChange () const
{
    return m_process->GetStopOnSharedLibraryEvents();
}

void
DynamicLoader::SetStopWhenImagesChange (bool stop)
{
    m_process->SetStopOnSharedLibraryEvents (stop);
}

