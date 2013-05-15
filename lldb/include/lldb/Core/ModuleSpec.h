//===-- ModuleSpec.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ModuleSpec_h_
#define liblldb_ModuleSpec_h_

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/UUID.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Target/PathMappingList.h"

namespace lldb_private {

class ModuleSpec
{
public:
    ModuleSpec () :
        m_file (),
        m_platform_file (),
        m_symbol_file (),
        m_arch (),
        m_uuid (),
        m_object_name (),
        m_object_offset (0),
        m_object_mod_time (),
        m_source_mappings ()
    {
    }

    ModuleSpec (const FileSpec &file_spec) :
        m_file (file_spec),
        m_platform_file (),
        m_symbol_file (),
        m_arch (),
        m_uuid (),
        m_object_name (),
        m_object_offset (0),
        m_object_mod_time (),
        m_source_mappings ()
    {
    }

    ModuleSpec (const FileSpec &file_spec, const ArchSpec &arch) :
        m_file (file_spec),
        m_platform_file (),
        m_symbol_file (),
        m_arch (arch),
        m_uuid (),
        m_object_name (),
        m_object_offset (0),
        m_object_mod_time (),
        m_source_mappings ()
    {
    }
    
    ModuleSpec (const ModuleSpec &rhs) :
        m_file (rhs.m_file),
        m_platform_file (rhs.m_platform_file),
        m_symbol_file (rhs.m_symbol_file),
        m_arch (rhs.m_arch),
        m_uuid (rhs.m_uuid),
        m_object_name (rhs.m_object_name),
        m_object_offset (rhs.m_object_offset),
        m_object_mod_time (rhs.m_object_mod_time),
        m_source_mappings (rhs.m_source_mappings)
    {
    }

    ModuleSpec &
    operator = (const ModuleSpec &rhs)
    {
        if (this != &rhs)
        {
            m_file = rhs.m_file;
            m_platform_file = rhs.m_platform_file;
            m_symbol_file = rhs.m_symbol_file;
            m_arch = rhs.m_arch;
            m_uuid = rhs.m_uuid;
            m_object_name = rhs.m_object_name;
            m_object_offset = rhs.m_object_offset;
            m_object_mod_time = rhs.m_object_mod_time;
            m_source_mappings = rhs.m_source_mappings;
        }
        return *this;
    }

    FileSpec *
    GetFileSpecPtr ()
    {
        if (m_file)
            return &m_file;
        return NULL;
    }

    const FileSpec *
    GetFileSpecPtr () const
    {
        if (m_file)
            return &m_file;
        return NULL;
    }
    
    FileSpec &
    GetFileSpec ()
    {
        return m_file;
    }
    const FileSpec &
    GetFileSpec () const
    {
        return m_file;
    }

    FileSpec *
    GetPlatformFileSpecPtr ()
    {
        if (m_platform_file)
            return &m_platform_file;
        return NULL;
    }

    const FileSpec *
    GetPlatformFileSpecPtr () const
    {
        if (m_platform_file)
            return &m_platform_file;
        return NULL;
    }

    FileSpec &
    GetPlatformFileSpec ()
    {
        return m_platform_file;
    }

    const FileSpec &
    GetPlatformFileSpec () const
    {
        return m_platform_file;
    }

    FileSpec *
    GetSymbolFileSpecPtr ()
    {
        if (m_symbol_file)
            return &m_symbol_file;
        return NULL;
    }
    
    const FileSpec *
    GetSymbolFileSpecPtr () const
    {
        if (m_symbol_file)
            return &m_symbol_file;
        return NULL;
    }
    
    FileSpec &
    GetSymbolFileSpec ()
    {
        return m_symbol_file;
    }
    
    const FileSpec &
    GetSymbolFileSpec () const
    {
        return m_symbol_file;
    }

    
    ArchSpec *
    GetArchitecturePtr ()
    {
        if (m_arch.IsValid())
            return &m_arch;
        return NULL;
    }
    
    const ArchSpec *
    GetArchitecturePtr () const
    {
        if (m_arch.IsValid())
            return &m_arch;
        return NULL;
    }
    
    ArchSpec &
    GetArchitecture ()
    {
        return m_arch;
    }
    
    const ArchSpec &
    GetArchitecture () const
    {
        return m_arch;
    }

    UUID *
    GetUUIDPtr ()
    {
        if (m_uuid.IsValid())
            return &m_uuid;
        return NULL;
    }
    
    const UUID *
    GetUUIDPtr () const
    {
        if (m_uuid.IsValid())
            return &m_uuid;
        return NULL;
    }
    
    UUID &
    GetUUID ()
    {
        return m_uuid;
    }
    
    const UUID &
    GetUUID () const
    {
        return m_uuid;
    }

    ConstString &
    GetObjectName ()
    {
        return m_object_name;
    }

    const ConstString &
    GetObjectName () const
    {
        return m_object_name;
    }

    uint64_t
    GetObjectOffset () const
    {
        return m_object_offset;
    }

    void
    SetObjectOffset (uint64_t object_offset)
    {
        m_object_offset = object_offset;
    }
    
    TimeValue &
    GetObjectModificationTime ()
    {
        return m_object_mod_time;
    }
    
    const TimeValue &
    GetObjectModificationTime () const
    {
        return m_object_mod_time;
    }

    PathMappingList &
    GetSourceMappingList () const
    {
        return m_source_mappings;
    }

    void
    Clear ()
    {
        m_file.Clear();
        m_platform_file.Clear();
        m_symbol_file.Clear();
        m_arch.Clear();
        m_uuid.Clear();
        m_object_name.Clear();
        m_object_offset = 0;
        m_source_mappings.Clear(false);
        m_object_mod_time.Clear();
    }

protected:
    FileSpec m_file;
    FileSpec m_platform_file;
    FileSpec m_symbol_file;
    ArchSpec m_arch;
    UUID m_uuid;
    ConstString m_object_name;
    uint64_t m_object_offset;
    TimeValue m_object_mod_time;
    mutable PathMappingList m_source_mappings;
};

class ModuleSpecList
{
public:
    ModuleSpecList () :
        m_specs(),
        m_mutex(Mutex::eMutexTypeRecursive) 
    {
    }

    ModuleSpecList (const ModuleSpecList &rhs) :
        m_specs(),
        m_mutex(Mutex::eMutexTypeRecursive)
    {
        Mutex::Locker lhs_locker(m_mutex);
        Mutex::Locker rhs_locker(rhs.m_mutex);
        m_specs = rhs.m_specs;
    }

    ~ModuleSpecList ()
    {
    }
    
    size_t
    GetSize() const
    {
        Mutex::Locker locker(m_mutex);
        return m_specs.size();
    }

    void
    Clear ()
    {
        Mutex::Locker locker(m_mutex);
        m_specs.clear();
    }

    void
    Append (const ModuleSpec &spec)
    {
        Mutex::Locker locker(m_mutex);
        m_specs.push_back (spec);
    }
    
    bool
    GetModuleSpecAtIndex (size_t i, ModuleSpec &module_spec) const
    {
        Mutex::Locker locker(m_mutex);
        if (i < m_specs.size())
        {
            module_spec = m_specs[i];
            return true;
        }
        module_spec.Clear();
        return false;
    }
    
    bool
    FindMatchingModuleSpec (const ModuleSpec &module_spec, ModuleSpec &match_module_spec) const
    {
        const FileSpec *file_ptr = module_spec.GetFileSpecPtr();
        const FileSpec *platform_file_ptr = module_spec.GetPlatformFileSpecPtr();
        const FileSpec *symbol_file_ptr = module_spec.GetSymbolFileSpecPtr();
        const ArchSpec *arch_ptr = module_spec.GetArchitecturePtr();
        const UUID *uuid_ptr = module_spec.GetUUIDPtr();
        const bool check_module_name = (bool)module_spec.GetObjectName();
        Mutex::Locker locker(m_mutex);
        for (auto spec: m_specs)
        {
            if (uuid_ptr && spec.GetUUID() != *uuid_ptr)
                continue;
            if (check_module_name && module_spec.GetObjectName() != spec.GetObjectName())
                continue;
            if (file_ptr && !FileSpec::Equal(*file_ptr, spec.GetFileSpec(), file_ptr->GetDirectory().IsEmpty() == false))
                continue;
            if (platform_file_ptr && !FileSpec::Equal(*platform_file_ptr, spec.GetFileSpec(), platform_file_ptr->GetDirectory().IsEmpty() == false))
                continue;
            if (symbol_file_ptr && !FileSpec::Equal(*symbol_file_ptr, spec.GetFileSpec(), symbol_file_ptr->GetDirectory().IsEmpty() == false))
                continue;
            if (arch_ptr && !spec.GetArchitecture().IsExactMatch(*arch_ptr))
                continue;
            match_module_spec = spec;
            return true;
        }
        
        // If there was an architecture, retry with a compatible arch
        if (arch_ptr)
        {
            for (auto spec: m_specs)
            {
                if (uuid_ptr && spec.GetUUID() != *uuid_ptr)
                    continue;
                if (check_module_name && module_spec.GetObjectName() != spec.GetObjectName())
                    continue;
                if (file_ptr && !FileSpec::Equal(*file_ptr, spec.GetFileSpec(), file_ptr->GetDirectory().IsEmpty() == false))
                    continue;
                if (platform_file_ptr && !FileSpec::Equal(*platform_file_ptr, spec.GetFileSpec(), platform_file_ptr->GetDirectory().IsEmpty() == false))
                    continue;
                if (symbol_file_ptr && !FileSpec::Equal(*symbol_file_ptr, spec.GetFileSpec(), symbol_file_ptr->GetDirectory().IsEmpty() == false))
                    continue;
                if (arch_ptr && !spec.GetArchitecture().IsCompatibleMatch(*arch_ptr))
                    continue;
                match_module_spec = spec;
                return true;
            }
        }
        match_module_spec.Clear();
        return false;
    }

protected:
    typedef std::vector<ModuleSpec> collection; ///< The module collection type.
    collection m_specs; ///< The collection of modules.
    mutable Mutex m_mutex;
};

} // namespace lldb_private

#endif  // liblldb_ModuleSpec_h_
