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

    PathMappingList &
    GetSourceMappingList () const
    {
        return m_source_mappings;
    }

protected:
    FileSpec m_file;
    FileSpec m_platform_file;
    FileSpec m_symbol_file;
    ArchSpec m_arch;
    UUID m_uuid;
    ConstString m_object_name;
    uint64_t m_object_offset;
    mutable PathMappingList m_source_mappings;
};

} // namespace lldb_private

#endif  // liblldb_ModuleSpec_h_
