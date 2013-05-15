//===-- SBModule.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBModule.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBSymbolContextList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;


SBModule::SBModule () :
    m_opaque_sp ()
{
}

SBModule::SBModule (const lldb::ModuleSP& module_sp) :
    m_opaque_sp (module_sp)
{
}

SBModule::SBModule(const SBModule &rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

SBModule::SBModule (lldb::SBProcess &process, lldb::addr_t header_addr) :
    m_opaque_sp ()
{
    ProcessSP process_sp (process.GetSP());
    if (process_sp)
    {
        m_opaque_sp = process_sp->ReadModuleFromMemory (FileSpec(), header_addr);
        if (m_opaque_sp)
        {
            Target &target = process_sp->GetTarget();
            bool changed = false;
            m_opaque_sp->SetLoadAddress(target, 0, changed);
            target.GetImages().Append(m_opaque_sp);
        }
    }
}

const SBModule &
SBModule::operator = (const SBModule &rhs)
{
    if (this != &rhs)
        m_opaque_sp = rhs.m_opaque_sp;
    return *this;
}

SBModule::~SBModule ()
{
}

bool
SBModule::IsValid () const
{
    return m_opaque_sp.get() != NULL;
}

void
SBModule::Clear()
{
    m_opaque_sp.reset();
}

SBFileSpec
SBModule::GetFileSpec () const
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBFileSpec file_spec;
    ModuleSP module_sp (GetSP ());
    if (module_sp)
        file_spec.SetFileSpec(module_sp->GetFileSpec());

    if (log)
    {
        log->Printf ("SBModule(%p)::GetFileSpec () => SBFileSpec(%p)", 
        module_sp.get(), file_spec.get());
    }

    return file_spec;
}

lldb::SBFileSpec
SBModule::GetPlatformFileSpec () const
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    SBFileSpec file_spec;
    ModuleSP module_sp (GetSP ());
    if (module_sp)
        file_spec.SetFileSpec(module_sp->GetPlatformFileSpec());
    
    if (log)
    {
        log->Printf ("SBModule(%p)::GetPlatformFileSpec () => SBFileSpec(%p)", 
                     module_sp.get(), file_spec.get());
    }
    
    return file_spec;
    
}

bool
SBModule::SetPlatformFileSpec (const lldb::SBFileSpec &platform_file)
{
    bool result = false;
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    ModuleSP module_sp (GetSP ());
    if (module_sp)
    {
        module_sp->SetPlatformFileSpec(*platform_file);
        result = true;
    }
    
    if (log)
    {
        log->Printf ("SBModule(%p)::SetPlatformFileSpec (SBFileSpec(%p (%s)) => %i", 
                     module_sp.get(), 
                     platform_file.get(),
                     platform_file->GetPath().c_str(),
                     result);
    }
    return result;
}



const uint8_t *
SBModule::GetUUIDBytes () const
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    const uint8_t *uuid_bytes = NULL;
    ModuleSP module_sp (GetSP ());
    if (module_sp)
        uuid_bytes = (const uint8_t *)module_sp->GetUUID().GetBytes();

    if (log)
    {
        if (uuid_bytes)
        {
            StreamString s;
            module_sp->GetUUID().Dump (&s);
            log->Printf ("SBModule(%p)::GetUUIDBytes () => %s", module_sp.get(), s.GetData());
        }
        else
            log->Printf ("SBModule(%p)::GetUUIDBytes () => NULL", module_sp.get());
    }
    return uuid_bytes;
}


const char *
SBModule::GetUUIDString () const
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    static char uuid_string_buffer[80];
    const char *uuid_c_string = NULL;
    std::string uuid_string;
    ModuleSP module_sp (GetSP ());
    if (module_sp)
        uuid_string = module_sp->GetUUID().GetAsString();

    if (!uuid_string.empty())
    {
        strncpy (uuid_string_buffer, uuid_string.c_str(), sizeof (uuid_string_buffer));
        uuid_c_string = uuid_string_buffer;
    }

    if (log)
    {
        if (!uuid_string.empty())
        {
            StreamString s;
            module_sp->GetUUID().Dump (&s);
            log->Printf ("SBModule(%p)::GetUUIDString () => %s", module_sp.get(), s.GetData());
        }
        else
            log->Printf ("SBModule(%p)::GetUUIDString () => NULL", module_sp.get());
    }
    return uuid_c_string;
}


bool
SBModule::operator == (const SBModule &rhs) const
{
    if (m_opaque_sp)
        return m_opaque_sp.get() == rhs.m_opaque_sp.get();
    return false;
}

bool
SBModule::operator != (const SBModule &rhs) const
{
    if (m_opaque_sp)
        return m_opaque_sp.get() != rhs.m_opaque_sp.get();
    return false;
}

ModuleSP
SBModule::GetSP () const
{
    return m_opaque_sp;
}

void
SBModule::SetSP (const ModuleSP &module_sp)
{
    m_opaque_sp = module_sp;
}

SBAddress
SBModule::ResolveFileAddress (lldb::addr_t vm_addr)
{
    lldb::SBAddress sb_addr;
    ModuleSP module_sp (GetSP ());
    if (module_sp)
    {
        Address addr;
        if (module_sp->ResolveFileAddress (vm_addr, addr))
            sb_addr.ref() = addr;
    }
    return sb_addr;
}

SBSymbolContext
SBModule::ResolveSymbolContextForAddress (const SBAddress& addr, uint32_t resolve_scope)
{
    SBSymbolContext sb_sc;
    ModuleSP module_sp (GetSP ());
    if (module_sp && addr.IsValid())
        module_sp->ResolveSymbolContextForAddress (addr.ref(), resolve_scope, *sb_sc);
    return sb_sc;
}

bool
SBModule::GetDescription (SBStream &description)
{
    Stream &strm = description.ref();

    ModuleSP module_sp (GetSP ());
    if (module_sp)
    {
        module_sp->GetDescription (&strm);
    }
    else
        strm.PutCString ("No value");

    return true;
}

uint32_t
SBModule::GetNumCompileUnits()
{
    ModuleSP module_sp (GetSP ());
    if (module_sp)
    {
        return module_sp->GetNumCompileUnits ();
    }
    return 0;
}

SBCompileUnit
SBModule::GetCompileUnitAtIndex (uint32_t index)
{
    SBCompileUnit sb_cu;
    ModuleSP module_sp (GetSP ());
    if (module_sp)
    {
        CompUnitSP cu_sp = module_sp->GetCompileUnitAtIndex (index);
        sb_cu.reset(cu_sp.get());
    }
    return sb_cu;
}

size_t
SBModule::GetNumSymbols ()
{
    ModuleSP module_sp (GetSP ());
    if (module_sp)
    {
        ObjectFile *obj_file = module_sp->GetObjectFile();
        if (obj_file)
        {
            Symtab *symtab = obj_file->GetSymtab();
            if (symtab)
                return symtab->GetNumSymbols();
        }
    }
    return 0;
}

SBSymbol
SBModule::GetSymbolAtIndex (size_t idx)
{
    SBSymbol sb_symbol;
    ModuleSP module_sp (GetSP ());
    if (module_sp)
    {
        ObjectFile *obj_file = module_sp->GetObjectFile();
        if (obj_file)
        {
            Symtab *symtab = obj_file->GetSymtab();
            if (symtab)
                sb_symbol.SetSymbol(symtab->SymbolAtIndex (idx));
        }
    }
    return sb_symbol;
}

lldb::SBSymbol
SBModule::FindSymbol (const char *name,
                      lldb::SymbolType symbol_type)
{
    SBSymbol sb_symbol;
    if (name && name[0])
    {
        ModuleSP module_sp (GetSP ());
        if (module_sp)
        {
            ObjectFile *obj_file = module_sp->GetObjectFile();
            if (obj_file)
            {
                Symtab *symtab = obj_file->GetSymtab();
                if (symtab)
                    sb_symbol.SetSymbol(symtab->FindFirstSymbolWithNameAndType(ConstString(name), symbol_type, Symtab::eDebugAny, Symtab::eVisibilityAny));
            }
        }
    }
    return sb_symbol;
}


lldb::SBSymbolContextList
SBModule::FindSymbols (const char *name, lldb::SymbolType symbol_type)
{
    SBSymbolContextList sb_sc_list;
    if (name && name[0])
    {
        ModuleSP module_sp (GetSP ());
        if (module_sp)
        {
            ObjectFile *obj_file = module_sp->GetObjectFile();
            if (obj_file)
            {
                Symtab *symtab = obj_file->GetSymtab();
                if (symtab)
                {
                    std::vector<uint32_t> matching_symbol_indexes;
                    const size_t num_matches = symtab->FindAllSymbolsWithNameAndType(ConstString(name), symbol_type, matching_symbol_indexes);
                    if (num_matches)
                    {
                        SymbolContext sc;
                        sc.module_sp = module_sp;
                        SymbolContextList &sc_list = *sb_sc_list;
                        for (size_t i=0; i<num_matches; ++i)
                        {
                            sc.symbol = symtab->SymbolAtIndex (matching_symbol_indexes[i]);
                            if (sc.symbol)
                                sc_list.Append(sc);
                        }
                    }
                }
            }
        }
    }
    return sb_sc_list;
    
}



size_t
SBModule::GetNumSections ()
{
    ModuleSP module_sp (GetSP ());
    if (module_sp)
    {
        ObjectFile *obj_file = module_sp->GetObjectFile();
        if (obj_file)
        {
            SectionList *section_list = obj_file->GetSectionList ();
            if (section_list)
                return section_list->GetSize();
        }
    }
    return 0;
}

SBSection
SBModule::GetSectionAtIndex (size_t idx)
{
    SBSection sb_section;
    ModuleSP module_sp (GetSP ());
    if (module_sp)
    {
        ObjectFile *obj_file = module_sp->GetObjectFile();
        if (obj_file)
        {
            SectionList *section_list = obj_file->GetSectionList ();

            if (section_list)
                sb_section.SetSP(section_list->GetSectionAtIndex (idx));
        }
    }
    return sb_section;
}

lldb::SBSymbolContextList
SBModule::FindFunctions (const char *name, 
                         uint32_t name_type_mask)
{
    lldb::SBSymbolContextList sb_sc_list;
    ModuleSP module_sp (GetSP ());
    if (name && module_sp)
    {
        const bool append = true;
        const bool symbols_ok = true;
        const bool inlines_ok = true;
        module_sp->FindFunctions (ConstString(name),
                                  NULL,
                                  name_type_mask, 
                                  symbols_ok,
                                  inlines_ok,
                                  append, 
                                  *sb_sc_list);
    }
    return sb_sc_list;
}


SBValueList
SBModule::FindGlobalVariables (SBTarget &target, const char *name, uint32_t max_matches)
{
    SBValueList sb_value_list;
    ModuleSP module_sp (GetSP ());
    if (name && module_sp)
    {
        VariableList variable_list;
        const uint32_t match_count = module_sp->FindGlobalVariables (ConstString (name),
                                                                     NULL,
                                                                     false, 
                                                                     max_matches,
                                                                     variable_list);

        if (match_count > 0)
        {
            for (uint32_t i=0; i<match_count; ++i)
            {
                lldb::ValueObjectSP valobj_sp;
                TargetSP target_sp (target.GetSP());
                valobj_sp = ValueObjectVariable::Create (target_sp.get(), variable_list.GetVariableAtIndex(i));
                if (valobj_sp)
                    sb_value_list.Append(SBValue(valobj_sp));
            }
        }
    }
    
    return sb_value_list;
}

lldb::SBValue
SBModule::FindFirstGlobalVariable (lldb::SBTarget &target, const char *name)
{
    SBValueList sb_value_list(FindGlobalVariables(target, name, 1));
    if (sb_value_list.IsValid() && sb_value_list.GetSize() > 0)
        return sb_value_list.GetValueAtIndex(0);
    return SBValue();
}

lldb::SBType
SBModule::FindFirstType (const char *name_cstr)
{
    SBType sb_type;
    ModuleSP module_sp (GetSP ());
    if (name_cstr && module_sp)
    {
        SymbolContext sc;
        const bool exact_match = false;
        ConstString name(name_cstr);

        sb_type = SBType (module_sp->FindFirstType(sc, name, exact_match));
        
        if (!sb_type.IsValid())
            sb_type = SBType (ClangASTType::GetBasicType (module_sp->GetClangASTContext().getASTContext(), name));
    }
    return sb_type;
}

lldb::SBType
SBModule::GetBasicType(lldb::BasicType type)
{
    ModuleSP module_sp (GetSP ());
    if (module_sp)
        return SBType (ClangASTType::GetBasicType (module_sp->GetClangASTContext().getASTContext(), type));
    return SBType();
}

lldb::SBTypeList
SBModule::FindTypes (const char *type)
{
    SBTypeList retval;
    
    ModuleSP module_sp (GetSP ());
    if (type && module_sp)
    {
        SymbolContext sc;
        TypeList type_list;
        const bool exact_match = false;
        ConstString name(type);
        const uint32_t num_matches = module_sp->FindTypes (sc,
                                                           name,
                                                           exact_match,
                                                           UINT32_MAX,
                                                           type_list);
        
        if (num_matches > 0)
        {
            for (size_t idx = 0; idx < num_matches; idx++)
            {
                TypeSP type_sp (type_list.GetTypeAtIndex(idx));
                if (type_sp)
                    retval.Append(SBType(type_sp));
            }
        }
        else
        {
            SBType sb_type(ClangASTType::GetBasicType (module_sp->GetClangASTContext().getASTContext(), name));
            if (sb_type.IsValid())
                retval.Append(sb_type);
        }
    }

    return retval;
}


SBSection
SBModule::FindSection (const char *sect_name)
{
    SBSection sb_section;
    
    ModuleSP module_sp (GetSP ());
    if (sect_name && module_sp)
    {
        ObjectFile *objfile = module_sp->GetObjectFile();
        if (objfile)
        {
            SectionList *section_list = objfile->GetSectionList();
            if (section_list)
            {
                ConstString const_sect_name(sect_name);
                SectionSP section_sp (section_list->FindSectionByName(const_sect_name));
                if (section_sp)
                {
                    sb_section.SetSP (section_sp);
                }
            }
        }
    }
    return sb_section;
}

lldb::ByteOrder
SBModule::GetByteOrder ()
{
    ModuleSP module_sp (GetSP ());
    if (module_sp)
        return module_sp->GetArchitecture().GetByteOrder();
    return eByteOrderInvalid;
}

const char *
SBModule::GetTriple ()
{
    ModuleSP module_sp (GetSP ());
    if (module_sp)
    {
        std::string triple (module_sp->GetArchitecture().GetTriple().str());
        // Unique the string so we don't run into ownership issues since
        // the const strings put the string into the string pool once and
        // the strings never comes out
        ConstString const_triple (triple.c_str());
        return const_triple.GetCString();
    }
    return NULL;
}

uint32_t
SBModule::GetAddressByteSize()
{
    ModuleSP module_sp (GetSP ());
    if (module_sp)
        return module_sp->GetArchitecture().GetAddressByteSize();
    return sizeof(void*);
}


uint32_t
SBModule::GetVersion (uint32_t *versions, uint32_t num_versions)
{
    ModuleSP module_sp (GetSP ());
    if (module_sp)
        return module_sp->GetVersion(versions, num_versions);
    else
    {
        if (versions && num_versions)
        {
            for (uint32_t i=0; i<num_versions; ++i)
                versions[i] = UINT32_MAX;
        }
        return 0;
    }
}

