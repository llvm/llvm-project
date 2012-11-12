//===-- ModuleList.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ModuleList.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/Symbols.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/VariableList.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ModuleList constructor
//----------------------------------------------------------------------
ModuleList::ModuleList() :
    m_modules(),
    m_modules_mutex (Mutex::eMutexTypeRecursive),
    m_notifier(NULL)
{
}

//----------------------------------------------------------------------
// Copy constructor
//----------------------------------------------------------------------
ModuleList::ModuleList(const ModuleList& rhs) :
    m_modules(),
    m_modules_mutex (Mutex::eMutexTypeRecursive)
{
    Mutex::Locker lhs_locker(m_modules_mutex);
    Mutex::Locker rhs_locker(rhs.m_modules_mutex);
    m_modules = rhs.m_modules;
}

ModuleList::ModuleList (ModuleList::Notifier* notifier) :
    m_modules(),
    m_modules_mutex (Mutex::eMutexTypeRecursive),
    m_notifier(notifier)
{
}

//----------------------------------------------------------------------
// Assignment operator
//----------------------------------------------------------------------
const ModuleList&
ModuleList::operator= (const ModuleList& rhs)
{
    if (this != &rhs)
    {
        Mutex::Locker lhs_locker(m_modules_mutex);
        Mutex::Locker rhs_locker(rhs.m_modules_mutex);
        m_modules = rhs.m_modules;
    }
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ModuleList::~ModuleList()
{
}

void
ModuleList::AppendImpl (const ModuleSP &module_sp, bool use_notifier)
{
    if (module_sp)
    {
        Mutex::Locker locker(m_modules_mutex);
        m_modules.push_back(module_sp);
        if (use_notifier && m_notifier)
            m_notifier->ModuleAdded(*this, module_sp);
    }
}

void
ModuleList::Append (const ModuleSP &module_sp)
{
    AppendImpl (module_sp);
}

void
ModuleList::ReplaceEquivalent (const ModuleSP &module_sp)
{
    if (module_sp)
    {
        Mutex::Locker locker(m_modules_mutex);

        // First remove any equivalent modules. Equivalent modules are modules
        // whose path, platform path and architecture match.
        ModuleSpec equivalent_module_spec (module_sp->GetFileSpec(), module_sp->GetArchitecture());
        equivalent_module_spec.GetPlatformFileSpec() = module_sp->GetPlatformFileSpec();

        size_t idx = 0;
        while (idx < m_modules.size())
        {
            ModuleSP module_sp (m_modules[idx]);
            if (module_sp->MatchesModuleSpec (equivalent_module_spec))
                RemoveImpl(m_modules.begin() + idx);
            else
                ++idx;
        }
        // Now add the new module to the list
        Append(module_sp);
    }
}

bool
ModuleList::AppendIfNeeded (const ModuleSP &module_sp)
{
    if (module_sp)
    {
        Mutex::Locker locker(m_modules_mutex);
        collection::iterator pos, end = m_modules.end();
        for (pos = m_modules.begin(); pos != end; ++pos)
        {
            if (pos->get() == module_sp.get())
                return false; // Already in the list
        }
        // Only push module_sp on the list if it wasn't already in there.
        Append(module_sp);
        return true;
    }
    return false;
}

void
ModuleList::Append (const ModuleList& module_list)
{
    for (auto pos : module_list.m_modules)
        Append(pos);
}

bool
ModuleList::AppendIfNeeded (const ModuleList& module_list)
{
    bool any_in = false;
    for (auto pos : module_list.m_modules)
        any_in = AppendIfNeeded(pos) | any_in;
    return any_in;
}

bool
ModuleList::RemoveImpl (const ModuleSP &module_sp, bool use_notifier)
{
    if (module_sp)
    {
        Mutex::Locker locker(m_modules_mutex);
        collection::iterator pos, end = m_modules.end();
        for (pos = m_modules.begin(); pos != end; ++pos)
        {
            if (pos->get() == module_sp.get())
            {
                m_modules.erase (pos);
                if (use_notifier && m_notifier)
                    m_notifier->ModuleRemoved(*this, module_sp);
                return true;
            }
        }
    }
    return false;
}

ModuleList::collection::iterator
ModuleList::RemoveImpl (ModuleList::collection::iterator pos, bool use_notifier)
{
    ModuleSP module_sp(*pos);
    collection::iterator retval = m_modules.erase(pos);
    if (use_notifier && m_notifier)
        m_notifier->ModuleRemoved(*this, module_sp);
    return retval;
}

bool
ModuleList::Remove (const ModuleSP &module_sp)
{
    return RemoveImpl (module_sp);
}

bool
ModuleList::ReplaceModule (const lldb::ModuleSP &old_module_sp, const lldb::ModuleSP &new_module_sp)
{
    if (!RemoveImpl(old_module_sp, false))
        return false;
    AppendImpl (new_module_sp, false);
    if (m_notifier)
        m_notifier->ModuleUpdated(*this, old_module_sp,new_module_sp);
    return true;
}

bool
ModuleList::RemoveIfOrphaned (const Module *module_ptr)
{
    if (module_ptr)
    {
        Mutex::Locker locker(m_modules_mutex);
        collection::iterator pos, end = m_modules.end();
        for (pos = m_modules.begin(); pos != end; ++pos)
        {
            if (pos->get() == module_ptr)
            {
                if (pos->unique())
                {
                    pos = RemoveImpl(pos);
                    return true;
                }
                else
                    return false;
            }
        }
    }
    return false;
}

size_t
ModuleList::RemoveOrphans (bool mandatory)
{
    Mutex::Locker locker;
    
    if (mandatory)
    {
        locker.Lock (m_modules_mutex);
    }
    else
    {
        // Not mandatory, remove orphans if we can get the mutex
        if (!locker.TryLock(m_modules_mutex))
            return 0;
    }
    collection::iterator pos = m_modules.begin();
    size_t remove_count = 0;
    while (pos != m_modules.end())
    {
        if (pos->unique())
        {
            pos = RemoveImpl(pos);
            ++remove_count;
        }
        else
        {
            ++pos;
        }
    }
    return remove_count;
}

size_t
ModuleList::Remove (ModuleList &module_list)
{
    Mutex::Locker locker(m_modules_mutex);
    size_t num_removed = 0;
    collection::iterator pos, end = module_list.m_modules.end();
    for (pos = module_list.m_modules.begin(); pos != end; ++pos)
    {
        if (Remove (*pos))
            ++num_removed;
    }
    return num_removed;
}


void
ModuleList::Clear()
{
    ClearImpl();
}

void
ModuleList::Destroy()
{
    ClearImpl();
}

void
ModuleList::ClearImpl (bool use_notifier)
{
    Mutex::Locker locker(m_modules_mutex);
    if (use_notifier && m_notifier)
        m_notifier->WillClearList(*this);
    m_modules.clear();
}

Module*
ModuleList::GetModulePointerAtIndex (uint32_t idx) const
{
    Mutex::Locker locker(m_modules_mutex);
    return GetModulePointerAtIndexUnlocked(idx);
}

Module*
ModuleList::GetModulePointerAtIndexUnlocked (uint32_t idx) const
{
    if (idx < m_modules.size())
        return m_modules[idx].get();
    return NULL;
}

ModuleSP
ModuleList::GetModuleAtIndex(uint32_t idx) const
{
    Mutex::Locker locker(m_modules_mutex);
    return GetModuleAtIndexUnlocked(idx);
}

ModuleSP
ModuleList::GetModuleAtIndexUnlocked(uint32_t idx) const
{
    ModuleSP module_sp;
    if (idx < m_modules.size())
        module_sp = m_modules[idx];
    return module_sp;
}

uint32_t
ModuleList::FindFunctions (const ConstString &name, 
                           uint32_t name_type_mask, 
                           bool include_symbols,
                           bool include_inlines,
                           bool append, 
                           SymbolContextList &sc_list) const
{
    if (!append)
        sc_list.Clear();
    
    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        (*pos)->FindFunctions (name, NULL, name_type_mask, include_symbols, include_inlines, true, sc_list);
    }
    
    return sc_list.GetSize();
}

uint32_t
ModuleList::FindCompileUnits (const FileSpec &path, 
                              bool append, 
                              SymbolContextList &sc_list) const
{
    if (!append)
        sc_list.Clear();
    
    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        (*pos)->FindCompileUnits (path, true, sc_list);
    }
    
    return sc_list.GetSize();
}

uint32_t
ModuleList::FindGlobalVariables (const ConstString &name, 
                                 bool append, 
                                 uint32_t max_matches, 
                                 VariableList& variable_list) const
{
    size_t initial_size = variable_list.GetSize();
    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        (*pos)->FindGlobalVariables (name, NULL, append, max_matches, variable_list);
    }
    return variable_list.GetSize() - initial_size;
}


uint32_t
ModuleList::FindGlobalVariables (const RegularExpression& regex, 
                                 bool append, 
                                 uint32_t max_matches, 
                                 VariableList& variable_list) const
{
    size_t initial_size = variable_list.GetSize();
    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        (*pos)->FindGlobalVariables (regex, append, max_matches, variable_list);
    }
    return variable_list.GetSize() - initial_size;
}


size_t
ModuleList::FindSymbolsWithNameAndType (const ConstString &name, 
                                        SymbolType symbol_type, 
                                        SymbolContextList &sc_list,
                                        bool append) const
{
    Mutex::Locker locker(m_modules_mutex);
    if (!append)
        sc_list.Clear();
    size_t initial_size = sc_list.GetSize();
    
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
        (*pos)->FindSymbolsWithNameAndType (name, symbol_type, sc_list);
    return sc_list.GetSize() - initial_size;
}

size_t
ModuleList::FindSymbolsMatchingRegExAndType (const RegularExpression &regex, 
                                             lldb::SymbolType symbol_type, 
                                             SymbolContextList &sc_list,
                                             bool append) const
{
    Mutex::Locker locker(m_modules_mutex);
    if (!append)
        sc_list.Clear();
    size_t initial_size = sc_list.GetSize();
    
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
        (*pos)->FindSymbolsMatchingRegExAndType (regex, symbol_type, sc_list);
    return sc_list.GetSize() - initial_size;
}

size_t
ModuleList::FindModules (const ModuleSpec &module_spec, ModuleList& matching_module_list) const
{
    size_t existing_matches = matching_module_list.GetSize();

    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        ModuleSP module_sp(*pos);
        if (module_sp->MatchesModuleSpec (module_spec))
            matching_module_list.Append(module_sp);
    }
    return matching_module_list.GetSize() - existing_matches;
}

ModuleSP
ModuleList::FindModule (const Module *module_ptr) const
{
    ModuleSP module_sp;

    // Scope for "locker"
    {
        Mutex::Locker locker(m_modules_mutex);
        collection::const_iterator pos, end = m_modules.end();

        for (pos = m_modules.begin(); pos != end; ++pos)
        {
            if ((*pos).get() == module_ptr)
            {
                module_sp = (*pos);
                break;
            }
        }
    }
    return module_sp;

}

ModuleSP
ModuleList::FindModule (const UUID &uuid) const
{
    ModuleSP module_sp;
    
    if (uuid.IsValid())
    {
        Mutex::Locker locker(m_modules_mutex);
        collection::const_iterator pos, end = m_modules.end();
        
        for (pos = m_modules.begin(); pos != end; ++pos)
        {
            if ((*pos)->GetUUID() == uuid)
            {
                module_sp = (*pos);
                break;
            }
        }
    }
    return module_sp;
}


uint32_t
ModuleList::FindTypes (const SymbolContext& sc, const ConstString &name, bool name_is_fully_qualified, uint32_t max_matches, TypeList& types) const
{
    Mutex::Locker locker(m_modules_mutex);

    uint32_t total_matches = 0;
    collection::const_iterator pos, end = m_modules.end();
    if (sc.module_sp)
    {
        // The symbol context "sc" contains a module so we want to search that
        // one first if it is in our list...
        for (pos = m_modules.begin(); pos != end; ++pos)
        {
            if (sc.module_sp.get() == (*pos).get())
            {
                total_matches += (*pos)->FindTypes (sc, name, name_is_fully_qualified, max_matches, types);

                if (total_matches >= max_matches)
                    break;
            }
        }
    }
    
    if (total_matches < max_matches)
    {
        SymbolContext world_sc;
        for (pos = m_modules.begin(); pos != end; ++pos)
        {
            // Search the module if the module is not equal to the one in the symbol
            // context "sc". If "sc" contains a empty module shared pointer, then
            // the comparisong will always be true (valid_module_ptr != NULL).
            if (sc.module_sp.get() != (*pos).get())
                total_matches += (*pos)->FindTypes (world_sc, name, name_is_fully_qualified, max_matches, types);
            
            if (total_matches >= max_matches)
                break;
        }
    }
    
    return total_matches;
}

bool
ModuleList::FindSourceFile (const FileSpec &orig_spec, FileSpec &new_spec) const
{
    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        if ((*pos)->FindSourceFile (orig_spec, new_spec))
            return true;
    }
    return false;
}



ModuleSP
ModuleList::FindFirstModule (const ModuleSpec &module_spec) const
{
    ModuleSP module_sp;
    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        ModuleSP module_sp(*pos);
        if (module_sp->MatchesModuleSpec (module_spec))
            return module_sp;
    }
    return module_sp;

}

size_t
ModuleList::GetSize() const
{
    size_t size = 0;
    {
        Mutex::Locker locker(m_modules_mutex);
        size = m_modules.size();
    }
    return size;
}


void
ModuleList::Dump(Stream *s) const
{
//  s.Printf("%.*p: ", (int)sizeof(void*) * 2, this);
//  s.Indent();
//  s << "ModuleList\n";

    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        (*pos)->Dump(s);
    }
}

void
ModuleList::LogUUIDAndPaths (LogSP &log_sp, const char *prefix_cstr)
{
    if (log_sp)
    {   
        Mutex::Locker locker(m_modules_mutex);
        char uuid_cstr[256];
        collection::const_iterator pos, begin = m_modules.begin(), end = m_modules.end();
        for (pos = begin; pos != end; ++pos)
        {
            Module *module = pos->get();
            module->GetUUID().GetAsCString (uuid_cstr, sizeof(uuid_cstr));
            const FileSpec &module_file_spec = module->GetFileSpec();
            log_sp->Printf ("%s[%u] %s (%s) \"%s/%s\"", 
                            prefix_cstr ? prefix_cstr : "",
                            (uint32_t)std::distance (begin, pos),
                            uuid_cstr,
                            module->GetArchitecture().GetArchitectureName(),
                            module_file_spec.GetDirectory().GetCString(),
                            module_file_spec.GetFilename().GetCString());
        }
    }
}

bool
ModuleList::ResolveFileAddress (lldb::addr_t vm_addr, Address& so_addr) const
{
    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        if ((*pos)->ResolveFileAddress (vm_addr, so_addr))
            return true;
    }

    return false;
}

uint32_t
ModuleList::ResolveSymbolContextForAddress (const Address& so_addr, uint32_t resolve_scope, SymbolContext& sc) const
{
    // The address is already section offset so it has a module
    uint32_t resolved_flags = 0;
    ModuleSP module_sp (so_addr.GetModule());
    if (module_sp)
    {
        resolved_flags = module_sp->ResolveSymbolContextForAddress (so_addr,
                                                                    resolve_scope,
                                                                    sc);
    }
    else
    {
        Mutex::Locker locker(m_modules_mutex);
        collection::const_iterator pos, end = m_modules.end();
        for (pos = m_modules.begin(); pos != end; ++pos)
        {
            resolved_flags = (*pos)->ResolveSymbolContextForAddress (so_addr,
                                                                     resolve_scope,
                                                                     sc);
            if (resolved_flags != 0)
                break;
        }
    }

    return resolved_flags;
}

uint32_t
ModuleList::ResolveSymbolContextForFilePath 
(
    const char *file_path, 
    uint32_t line, 
    bool check_inlines, 
    uint32_t resolve_scope, 
    SymbolContextList& sc_list
)  const
{
    FileSpec file_spec(file_path, false);
    return ResolveSymbolContextsForFileSpec (file_spec, line, check_inlines, resolve_scope, sc_list);
}

uint32_t
ModuleList::ResolveSymbolContextsForFileSpec (const FileSpec &file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, SymbolContextList& sc_list) const
{
    Mutex::Locker locker(m_modules_mutex);
    collection::const_iterator pos, end = m_modules.end();
    for (pos = m_modules.begin(); pos != end; ++pos)
    {
        (*pos)->ResolveSymbolContextsForFileSpec (file_spec, line, check_inlines, resolve_scope, sc_list);
    }

    return sc_list.GetSize();
}

uint32_t
ModuleList::GetIndexForModule (const Module *module) const
{
    if (module)
    {
        Mutex::Locker locker(m_modules_mutex);
        collection::const_iterator pos;
        collection::const_iterator begin = m_modules.begin();
        collection::const_iterator end = m_modules.end();
        for (pos = begin; pos != end; ++pos)
        {
            if ((*pos).get() == module)
                return std::distance (begin, pos);
        }
    }
    return LLDB_INVALID_INDEX32;
}

static ModuleList &
GetSharedModuleList ()
{
    // NOTE: Intentionally leak the module list so a program doesn't have to
    // cleanup all modules and object files as it exits. This just wastes time
    // doing a bunch of cleanup that isn't required.
    static ModuleList *g_shared_module_list = NULL;
    if (g_shared_module_list == NULL)
        g_shared_module_list = new ModuleList(); // <--- Intentional leak!!!
    
    return *g_shared_module_list;
}

bool
ModuleList::ModuleIsInCache (const Module *module_ptr)
{
    if (module_ptr)
    {
        ModuleList &shared_module_list = GetSharedModuleList ();
        return shared_module_list.FindModule (module_ptr).get() != NULL;
    }
    return false;
}

size_t
ModuleList::FindSharedModules (const ModuleSpec &module_spec, ModuleList &matching_module_list)
{
    return GetSharedModuleList ().FindModules (module_spec, matching_module_list);
}

uint32_t
ModuleList::RemoveOrphanSharedModules (bool mandatory)
{
    return GetSharedModuleList ().RemoveOrphans(mandatory);
}

Error
ModuleList::GetSharedModule
(
    const ModuleSpec &module_spec,
    ModuleSP &module_sp,
    const FileSpecList *module_search_paths_ptr,
    ModuleSP *old_module_sp_ptr,
    bool *did_create_ptr,
    bool always_create
)
{
    ModuleList &shared_module_list = GetSharedModuleList ();
    Mutex::Locker locker(shared_module_list.m_modules_mutex);
    char path[PATH_MAX];
    char uuid_cstr[64];

    Error error;

    module_sp.reset();

    if (did_create_ptr)
        *did_create_ptr = false;
    if (old_module_sp_ptr)
        old_module_sp_ptr->reset();

    const UUID *uuid_ptr = module_spec.GetUUIDPtr();
    const FileSpec &module_file_spec = module_spec.GetFileSpec();
    const ArchSpec &arch = module_spec.GetArchitecture();

    // Make sure no one else can try and get or create a module while this
    // function is actively working on it by doing an extra lock on the
    // global mutex list.
    if (always_create == false)
    {
        ModuleList matching_module_list;
        const size_t num_matching_modules = shared_module_list.FindModules (module_spec, matching_module_list);
        if (num_matching_modules > 0)
        {
            for (uint32_t module_idx = 0; module_idx < num_matching_modules; ++module_idx)
            {
                module_sp = matching_module_list.GetModuleAtIndex(module_idx);
                
                // Make sure the file for the module hasn't been modified
                if (module_sp->FileHasChanged())
                {
                    if (old_module_sp_ptr && !old_module_sp_ptr->get())
                        *old_module_sp_ptr = module_sp;

                    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_MODULES));
                    if (log)
                        log->Printf("module changed: %p, removing from global module list", module_sp.get());

                    shared_module_list.Remove (module_sp);
                    module_sp.reset();
                }
                else
                {
                    // The module matches and the module was not modified from
                    // when it was last loaded.
                    return error;
                }
            }
        }
    }

    if (module_sp)
        return error;
    else
    {
        module_sp.reset (new Module (module_spec));
        // Make sure there are a module and an object file since we can specify
        // a valid file path with an architecture that might not be in that file.
        // By getting the object file we can guarantee that the architecture matches
        if (module_sp)
        {
            if (module_sp->GetObjectFile())
            {
                // If we get in here we got the correct arch, now we just need
                // to verify the UUID if one was given
                if (uuid_ptr && *uuid_ptr != module_sp->GetUUID())
                    module_sp.reset();
                else
                {
                    if (did_create_ptr)
                        *did_create_ptr = true;
                    
                    shared_module_list.ReplaceEquivalent(module_sp);
                    return error;
                }
            }
            else
                module_sp.reset();
        }
    }

    // Either the file didn't exist where at the path, or no path was given, so
    // we now have to use more extreme measures to try and find the appropriate
    // module.

    // Fixup the incoming path in case the path points to a valid file, yet
    // the arch or UUID (if one was passed in) don't match.
    FileSpec file_spec = Symbols::LocateExecutableObjectFile (module_spec);

    // Don't look for the file if it appears to be the same one we already
    // checked for above...
    if (file_spec != module_file_spec)
    {
        if (!file_spec.Exists())
        {
            file_spec.GetPath(path, sizeof(path));
            if (path[0] == '\0')
                module_file_spec.GetPath(path, sizeof(path));
            if (file_spec.Exists())
            {
                if (uuid_ptr && uuid_ptr->IsValid())
                    uuid_ptr->GetAsCString(uuid_cstr, sizeof (uuid_cstr));
                else
                    uuid_cstr[0] = '\0';


                if (arch.IsValid())
                {
                    if (uuid_cstr[0])
                        error.SetErrorStringWithFormat("'%s' does not contain the %s architecture and UUID %s", path, arch.GetArchitectureName(), uuid_cstr);
                    else
                        error.SetErrorStringWithFormat("'%s' does not contain the %s architecture.", path, arch.GetArchitectureName());
                }
            }
            else
            {
                error.SetErrorStringWithFormat("'%s' does not exist", path);
            }
            if (error.Fail())
                module_sp.reset();
            return error;
        }


        // Make sure no one else can try and get or create a module while this
        // function is actively working on it by doing an extra lock on the
        // global mutex list.
        ModuleSpec platform_module_spec(module_spec);
        platform_module_spec.GetFileSpec() = file_spec;
        platform_module_spec.GetPlatformFileSpec() = file_spec;
        ModuleList matching_module_list;
        if (shared_module_list.FindModules (platform_module_spec, matching_module_list) > 0)
        {
            module_sp = matching_module_list.GetModuleAtIndex(0);

            // If we didn't have a UUID in mind when looking for the object file,
            // then we should make sure the modification time hasn't changed!
            if (platform_module_spec.GetUUIDPtr() == NULL)
            {
                TimeValue file_spec_mod_time(file_spec.GetModificationTime());
                if (file_spec_mod_time.IsValid())
                {
                    if (file_spec_mod_time != module_sp->GetModificationTime())
                    {
                        if (old_module_sp_ptr)
                            *old_module_sp_ptr = module_sp;
                        shared_module_list.Remove (module_sp);
                        module_sp.reset();
                    }
                }
            }
        }

        if (module_sp.get() == NULL)
        {
            module_sp.reset (new Module (platform_module_spec));
            // Make sure there are a module and an object file since we can specify
            // a valid file path with an architecture that might not be in that file.
            // By getting the object file we can guarantee that the architecture matches
            if (module_sp && module_sp->GetObjectFile())
            {
                if (did_create_ptr)
                    *did_create_ptr = true;

                shared_module_list.ReplaceEquivalent(module_sp);
            }
            else
            {
                file_spec.GetPath(path, sizeof(path));

                if (file_spec)
                {
                    if (arch.IsValid())
                        error.SetErrorStringWithFormat("unable to open %s architecture in '%s'", arch.GetArchitectureName(), path);
                    else
                        error.SetErrorStringWithFormat("unable to open '%s'", path);
                }
                else
                {
                    if (uuid_ptr && uuid_ptr->IsValid())
                        uuid_ptr->GetAsCString(uuid_cstr, sizeof (uuid_cstr));
                    else
                        uuid_cstr[0] = '\0';

                    if (uuid_cstr[0])
                        error.SetErrorStringWithFormat("cannot locate a module for UUID '%s'", uuid_cstr);
                    else
                        error.SetErrorStringWithFormat("cannot locate a module");
                }
            }
        }
    }

    return error;
}

bool
ModuleList::RemoveSharedModule (lldb::ModuleSP &module_sp)
{
    return GetSharedModuleList ().Remove (module_sp);
}

bool
ModuleList::RemoveSharedModuleIfOrphaned (const Module *module_ptr)
{
    return GetSharedModuleList ().RemoveIfOrphaned (module_ptr);
}


