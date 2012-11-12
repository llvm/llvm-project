//===-- Module.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/Symbols.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

// Shared pointers to modules track module lifetimes in
// targets and in the global module, but this collection
// will track all module objects that are still alive
typedef std::vector<Module *> ModuleCollection;

static ModuleCollection &
GetModuleCollection()
{
    // This module collection needs to live past any module, so we could either make it a
    // shared pointer in each module or just leak is.  Since it is only an empty vector by
    // the time all the modules have gone away, we just leak it for now.  If we decide this 
    // is a big problem we can introduce a Finalize method that will tear everything down in
    // a predictable order.
    
    static ModuleCollection *g_module_collection = NULL;
    if (g_module_collection == NULL)
        g_module_collection = new ModuleCollection();
        
    return *g_module_collection;
}

Mutex *
Module::GetAllocationModuleCollectionMutex()
{
    // NOTE: The mutex below must be leaked since the global module list in
    // the ModuleList class will get torn at some point, and we can't know
    // if it will tear itself down before the "g_module_collection_mutex" below
    // will. So we leak a Mutex object below to safeguard against that

    static Mutex *g_module_collection_mutex = NULL;
    if (g_module_collection_mutex == NULL)
        g_module_collection_mutex = new Mutex (Mutex::eMutexTypeRecursive); // NOTE: known leak
    return g_module_collection_mutex;
}

size_t
Module::GetNumberAllocatedModules ()
{
    Mutex::Locker locker (GetAllocationModuleCollectionMutex());
    return GetModuleCollection().size();
}

Module *
Module::GetAllocatedModuleAtIndex (size_t idx)
{
    Mutex::Locker locker (GetAllocationModuleCollectionMutex());
    ModuleCollection &modules = GetModuleCollection();
    if (idx < modules.size())
        return modules[idx];
    return NULL;
}
#if 0

// These functions help us to determine if modules are still loaded, yet don't require that
// you have a command interpreter and can easily be called from an external debugger.
namespace lldb {

    void
    ClearModuleInfo (void)
    {
        const bool mandatory = true;
        ModuleList::RemoveOrphanSharedModules(mandatory);
    }
    
    void
    DumpModuleInfo (void)
    {
        Mutex::Locker locker (Module::GetAllocationModuleCollectionMutex());
        ModuleCollection &modules = GetModuleCollection();
        const size_t count = modules.size();
        printf ("%s: %llu modules:\n", __PRETTY_FUNCTION__, (uint64_t)count);
        for (size_t i=0; i<count; ++i)
        {
            
            StreamString strm;
            Module *module = modules[i];
            const bool in_shared_module_list = ModuleList::ModuleIsInCache (module);
            module->GetDescription(&strm, eDescriptionLevelFull);
            printf ("%p: shared = %i, ref_count = %3u, module = %s\n", 
                    module, 
                    in_shared_module_list,
                    (uint32_t)module->use_count(), 
                    strm.GetString().c_str());
        }
    }
}

#endif

Module::Module (const ModuleSpec &module_spec) :
    m_mutex (Mutex::eMutexTypeRecursive),
    m_mod_time (module_spec.GetFileSpec().GetModificationTime()),
    m_arch (module_spec.GetArchitecture()),
    m_uuid (),
    m_file (module_spec.GetFileSpec()),
    m_platform_file(module_spec.GetPlatformFileSpec()),
    m_symfile_spec (module_spec.GetSymbolFileSpec()),
    m_object_name (module_spec.GetObjectName()),
    m_object_offset (module_spec.GetObjectOffset()),
    m_objfile_sp (),
    m_symfile_ap (),
    m_ast (),
    m_source_mappings (),
    m_did_load_objfile (false),
    m_did_load_symbol_vendor (false),
    m_did_parse_uuid (false),
    m_did_init_ast (false),
    m_is_dynamic_loader_module (false),
    m_file_has_changed (false),
    m_first_file_changed_log (false)
{
    // Scope for locker below...
    {
        Mutex::Locker locker (GetAllocationModuleCollectionMutex());
        GetModuleCollection().push_back(this);
    }
    
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_OBJECT|LIBLLDB_LOG_MODULES));
    if (log)
        log->Printf ("%p Module::Module((%s) '%s/%s%s%s%s')",
                     this,
                     m_arch.GetArchitectureName(),
                     m_file.GetDirectory().AsCString(""),
                     m_file.GetFilename().AsCString(""),
                     m_object_name.IsEmpty() ? "" : "(",
                     m_object_name.IsEmpty() ? "" : m_object_name.AsCString(""),
                     m_object_name.IsEmpty() ? "" : ")");
}

Module::Module(const FileSpec& file_spec, 
               const ArchSpec& arch, 
               const ConstString *object_name, 
               off_t object_offset) :
    m_mutex (Mutex::eMutexTypeRecursive),
    m_mod_time (file_spec.GetModificationTime()),
    m_arch (arch),
    m_uuid (),
    m_file (file_spec),
    m_platform_file(),
    m_symfile_spec (),
    m_object_name (),
    m_object_offset (object_offset),
    m_objfile_sp (),
    m_symfile_ap (),
    m_ast (),
    m_source_mappings (),
    m_did_load_objfile (false),
    m_did_load_symbol_vendor (false),
    m_did_parse_uuid (false),
    m_did_init_ast (false),
    m_is_dynamic_loader_module (false),
    m_file_has_changed (false),
    m_first_file_changed_log (false)
{
    // Scope for locker below...
    {
        Mutex::Locker locker (GetAllocationModuleCollectionMutex());
        GetModuleCollection().push_back(this);
    }

    if (object_name)
        m_object_name = *object_name;
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_OBJECT|LIBLLDB_LOG_MODULES));
    if (log)
        log->Printf ("%p Module::Module((%s) '%s/%s%s%s%s')",
                     this,
                     m_arch.GetArchitectureName(),
                     m_file.GetDirectory().AsCString(""),
                     m_file.GetFilename().AsCString(""),
                     m_object_name.IsEmpty() ? "" : "(",
                     m_object_name.IsEmpty() ? "" : m_object_name.AsCString(""),
                     m_object_name.IsEmpty() ? "" : ")");
}

Module::~Module()
{
    // Scope for locker below...
    {
        Mutex::Locker locker (GetAllocationModuleCollectionMutex());
        ModuleCollection &modules = GetModuleCollection();
        ModuleCollection::iterator end = modules.end();
        ModuleCollection::iterator pos = std::find(modules.begin(), end, this);
        assert (pos != end);
        modules.erase(pos);
    }
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_OBJECT|LIBLLDB_LOG_MODULES));
    if (log)
        log->Printf ("%p Module::~Module((%s) '%s/%s%s%s%s')",
                     this,
                     m_arch.GetArchitectureName(),
                     m_file.GetDirectory().AsCString(""),
                     m_file.GetFilename().AsCString(""),
                     m_object_name.IsEmpty() ? "" : "(",
                     m_object_name.IsEmpty() ? "" : m_object_name.AsCString(""),
                     m_object_name.IsEmpty() ? "" : ")");
    // Release any auto pointers before we start tearing down our member 
    // variables since the object file and symbol files might need to make
    // function calls back into this module object. The ordering is important
    // here because symbol files can require the module object file. So we tear
    // down the symbol file first, then the object file.
    m_symfile_ap.reset();
    m_objfile_sp.reset();
}

ObjectFile *
Module::GetMemoryObjectFile (const lldb::ProcessSP &process_sp, lldb::addr_t header_addr, Error &error)
{
    if (m_objfile_sp)
    {
        error.SetErrorString ("object file already exists");
    }
    else
    {
        Mutex::Locker locker (m_mutex);
        if (process_sp)
        {
            m_did_load_objfile = true;
            std::auto_ptr<DataBufferHeap> data_ap (new DataBufferHeap (512, 0));
            Error readmem_error;
            const size_t bytes_read = process_sp->ReadMemory (header_addr, 
                                                              data_ap->GetBytes(), 
                                                              data_ap->GetByteSize(), 
                                                              readmem_error);
            if (bytes_read == 512)
            {
                DataBufferSP data_sp(data_ap.release());
                m_objfile_sp = ObjectFile::FindPlugin(shared_from_this(), process_sp, header_addr, data_sp);
                if (m_objfile_sp)
                {
                    StreamString s;
                    s.Printf("0x%16.16llx", header_addr);
                    m_object_name.SetCString (s.GetData());

                    // Once we get the object file, update our module with the object file's
                    // architecture since it might differ in vendor/os if some parts were
                    // unknown.
                    m_objfile_sp->GetArchitecture (m_arch);
                }
                else
                {
                    error.SetErrorString ("unable to find suitable object file plug-in");
                }
            }
            else
            {
                error.SetErrorStringWithFormat ("unable to read header from memory: %s", readmem_error.AsCString());
            }
        }
        else
        {
            error.SetErrorString ("invalid process");
        }
    }
    return m_objfile_sp.get();
}


const lldb_private::UUID&
Module::GetUUID()
{
    Mutex::Locker locker (m_mutex);
    if (m_did_parse_uuid == false)
    {
        ObjectFile * obj_file = GetObjectFile ();

        if (obj_file != NULL)
        {
            obj_file->GetUUID(&m_uuid);
            m_did_parse_uuid = true;
        }
    }
    return m_uuid;
}

ClangASTContext &
Module::GetClangASTContext ()
{
    Mutex::Locker locker (m_mutex);
    if (m_did_init_ast == false)
    {
        ObjectFile * objfile = GetObjectFile();
        ArchSpec object_arch;
        if (objfile && objfile->GetArchitecture(object_arch))
        {
            m_did_init_ast = true;

            // LLVM wants this to be set to iOS or MacOSX; if we're working on
            // a bare-boards type image, change the triple for llvm's benefit.
            if (object_arch.GetTriple().getVendor() == llvm::Triple::Apple 
                && object_arch.GetTriple().getOS() == llvm::Triple::UnknownOS)
            {
                if (object_arch.GetTriple().getArch() == llvm::Triple::arm || 
                    object_arch.GetTriple().getArch() == llvm::Triple::thumb)
                {
                    object_arch.GetTriple().setOS(llvm::Triple::IOS);
                }
                else
                {
                    object_arch.GetTriple().setOS(llvm::Triple::MacOSX);
                }
            }
            m_ast.SetArchitecture (object_arch);
        }
    }
    return m_ast;
}

void
Module::ParseAllDebugSymbols()
{
    Mutex::Locker locker (m_mutex);
    uint32_t num_comp_units = GetNumCompileUnits();
    if (num_comp_units == 0)
        return;

    SymbolContext sc;
    sc.module_sp = shared_from_this();
    uint32_t cu_idx;
    SymbolVendor *symbols = GetSymbolVendor ();

    for (cu_idx = 0; cu_idx < num_comp_units; cu_idx++)
    {
        sc.comp_unit = symbols->GetCompileUnitAtIndex(cu_idx).get();
        if (sc.comp_unit)
        {
            sc.function = NULL;
            symbols->ParseVariablesForContext(sc);

            symbols->ParseCompileUnitFunctions(sc);

            uint32_t func_idx;
            for (func_idx = 0; (sc.function = sc.comp_unit->GetFunctionAtIndex(func_idx).get()) != NULL; ++func_idx)
            {
                symbols->ParseFunctionBlocks(sc);

                // Parse the variables for this function and all its blocks
                symbols->ParseVariablesForContext(sc);
            }


            // Parse all types for this compile unit
            sc.function = NULL;
            symbols->ParseTypes(sc);
        }
    }
}

void
Module::CalculateSymbolContext(SymbolContext* sc)
{
    sc->module_sp = shared_from_this();
}

ModuleSP
Module::CalculateSymbolContextModule ()
{
    return shared_from_this();
}

void
Module::DumpSymbolContext(Stream *s)
{
    s->Printf(", Module{%p}", this);
}

uint32_t
Module::GetNumCompileUnits()
{
    Mutex::Locker locker (m_mutex);
    Timer scoped_timer(__PRETTY_FUNCTION__, "Module::GetNumCompileUnits (module = %p)", this);
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        return symbols->GetNumCompileUnits();
    return 0;
}

CompUnitSP
Module::GetCompileUnitAtIndex (uint32_t index)
{
    Mutex::Locker locker (m_mutex);
    uint32_t num_comp_units = GetNumCompileUnits ();
    CompUnitSP cu_sp;

    if (index < num_comp_units)
    {
        SymbolVendor *symbols = GetSymbolVendor ();
        if (symbols)
            cu_sp = symbols->GetCompileUnitAtIndex(index);
    }
    return cu_sp;
}

bool
Module::ResolveFileAddress (lldb::addr_t vm_addr, Address& so_addr)
{
    Mutex::Locker locker (m_mutex);
    Timer scoped_timer(__PRETTY_FUNCTION__, "Module::ResolveFileAddress (vm_addr = 0x%llx)", vm_addr);
    ObjectFile* ofile = GetObjectFile();
    if (ofile)
        return so_addr.ResolveAddressUsingFileSections(vm_addr, ofile->GetSectionList());
    return false;
}

uint32_t
Module::ResolveSymbolContextForAddress (const Address& so_addr, uint32_t resolve_scope, SymbolContext& sc)
{
    Mutex::Locker locker (m_mutex);
    uint32_t resolved_flags = 0;

    // Clear the result symbol context in case we don't find anything
    sc.Clear();

    // Get the section from the section/offset address.
    SectionSP section_sp (so_addr.GetSection());

    // Make sure the section matches this module before we try and match anything
    if (section_sp && section_sp->GetModule().get() == this)
    {
        // If the section offset based address resolved itself, then this
        // is the right module.
        sc.module_sp = shared_from_this();
        resolved_flags |= eSymbolContextModule;

        // Resolve the compile unit, function, block, line table or line
        // entry if requested.
        if (resolve_scope & eSymbolContextCompUnit    ||
            resolve_scope & eSymbolContextFunction    ||
            resolve_scope & eSymbolContextBlock       ||
            resolve_scope & eSymbolContextLineEntry   )
        {
            SymbolVendor *symbols = GetSymbolVendor ();
            if (symbols)
                resolved_flags |= symbols->ResolveSymbolContext (so_addr, resolve_scope, sc);
        }

        // Resolve the symbol if requested, but don't re-look it up if we've already found it.
        if (resolve_scope & eSymbolContextSymbol && !(resolved_flags & eSymbolContextSymbol))
        {
            ObjectFile* ofile = GetObjectFile();
            if (ofile)
            {
                Symtab *symtab = ofile->GetSymtab();
                if (symtab)
                {
                    if (so_addr.IsSectionOffset())
                    {
                        sc.symbol = symtab->FindSymbolContainingFileAddress(so_addr.GetFileAddress());
                        if (sc.symbol)
                            resolved_flags |= eSymbolContextSymbol;
                    }
                }
            }
        }
    }
    return resolved_flags;
}

uint32_t
Module::ResolveSymbolContextForFilePath 
(
    const char *file_path, 
    uint32_t line, 
    bool check_inlines, 
    uint32_t resolve_scope, 
    SymbolContextList& sc_list
)
{
    FileSpec file_spec(file_path, false);
    return ResolveSymbolContextsForFileSpec (file_spec, line, check_inlines, resolve_scope, sc_list);
}

uint32_t
Module::ResolveSymbolContextsForFileSpec (const FileSpec &file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, SymbolContextList& sc_list)
{
    Mutex::Locker locker (m_mutex);
    Timer scoped_timer(__PRETTY_FUNCTION__,
                       "Module::ResolveSymbolContextForFilePath (%s%s%s:%u, check_inlines = %s, resolve_scope = 0x%8.8x)",
                       file_spec.GetDirectory().AsCString(""),
                       file_spec.GetDirectory() ? "/" : "",
                       file_spec.GetFilename().AsCString(""),
                       line,
                       check_inlines ? "yes" : "no",
                       resolve_scope);

    const uint32_t initial_count = sc_list.GetSize();

    SymbolVendor *symbols = GetSymbolVendor  ();
    if (symbols)
        symbols->ResolveSymbolContext (file_spec, line, check_inlines, resolve_scope, sc_list);

    return sc_list.GetSize() - initial_count;
}


uint32_t
Module::FindGlobalVariables(const ConstString &name, const ClangNamespaceDecl *namespace_decl, bool append, uint32_t max_matches, VariableList& variables)
{
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        return symbols->FindGlobalVariables(name, namespace_decl, append, max_matches, variables);
    return 0;
}
uint32_t
Module::FindGlobalVariables(const RegularExpression& regex, bool append, uint32_t max_matches, VariableList& variables)
{
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        return symbols->FindGlobalVariables(regex, append, max_matches, variables);
    return 0;
}

uint32_t
Module::FindCompileUnits (const FileSpec &path,
                          bool append,
                          SymbolContextList &sc_list)
{
    if (!append)
        sc_list.Clear();
    
    const uint32_t start_size = sc_list.GetSize();
    const uint32_t num_compile_units = GetNumCompileUnits();
    SymbolContext sc;
    sc.module_sp = shared_from_this();
    const bool compare_directory = path.GetDirectory();
    for (uint32_t i=0; i<num_compile_units; ++i)
    {
        sc.comp_unit = GetCompileUnitAtIndex(i).get();
        if (sc.comp_unit)
        {
            if (FileSpec::Equal (*sc.comp_unit, path, compare_directory))
                sc_list.Append(sc);
        }
    }
    return sc_list.GetSize() - start_size;
}

uint32_t
Module::FindFunctions (const ConstString &name,
                       const ClangNamespaceDecl *namespace_decl,
                       uint32_t name_type_mask, 
                       bool include_symbols,
                       bool include_inlines,
                       bool append, 
                       SymbolContextList& sc_list)
{
    if (!append)
        sc_list.Clear();

    const uint32_t start_size = sc_list.GetSize();

    // Find all the functions (not symbols, but debug information functions...
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        symbols->FindFunctions(name, namespace_decl, name_type_mask, include_inlines, append, sc_list);

    // Now check our symbol table for symbols that are code symbols if requested
    if (include_symbols)
    {
        ObjectFile *objfile = GetObjectFile();
        if (objfile)
        {
            Symtab *symtab = objfile->GetSymtab();
            if (symtab)
            {
                std::vector<uint32_t> symbol_indexes;
                symtab->FindAllSymbolsWithNameAndType (name, eSymbolTypeCode, Symtab::eDebugAny, Symtab::eVisibilityAny, symbol_indexes);
                const uint32_t num_matches = symbol_indexes.size();
                if (num_matches)
                {
                    const bool merge_symbol_into_function = true;
                    SymbolContext sc(this);
                    for (uint32_t i=0; i<num_matches; i++)
                    {
                        sc.symbol = symtab->SymbolAtIndex(symbol_indexes[i]);
                        sc_list.AppendIfUnique (sc, merge_symbol_into_function);
                    }
                }
            }
        }
    }
    return sc_list.GetSize() - start_size;
}

uint32_t
Module::FindFunctions (const RegularExpression& regex, 
                       bool include_symbols,
                       bool include_inlines,
                       bool append, 
                       SymbolContextList& sc_list)
{
    if (!append)
        sc_list.Clear();
    
    const uint32_t start_size = sc_list.GetSize();
    
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        symbols->FindFunctions(regex, include_inlines, append, sc_list);
    // Now check our symbol table for symbols that are code symbols if requested
    if (include_symbols)
    {
        ObjectFile *objfile = GetObjectFile();
        if (objfile)
        {
            Symtab *symtab = objfile->GetSymtab();
            if (symtab)
            {
                std::vector<uint32_t> symbol_indexes;
                symtab->AppendSymbolIndexesMatchingRegExAndType (regex, eSymbolTypeCode, Symtab::eDebugAny, Symtab::eVisibilityAny, symbol_indexes);
                const uint32_t num_matches = symbol_indexes.size();
                if (num_matches)
                {
                    const bool merge_symbol_into_function = true;
                    SymbolContext sc(this);
                    for (uint32_t i=0; i<num_matches; i++)
                    {
                        sc.symbol = symtab->SymbolAtIndex(symbol_indexes[i]);
                        sc_list.AppendIfUnique (sc, merge_symbol_into_function);
                    }
                }
            }
        }
    }
    return sc_list.GetSize() - start_size;
}

uint32_t
Module::FindTypes_Impl (const SymbolContext& sc,
                        const ConstString &name,
                        const ClangNamespaceDecl *namespace_decl,
                        bool append,
                        uint32_t max_matches,
                        TypeList& types)
{
    Timer scoped_timer(__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
    if (sc.module_sp.get() == NULL || sc.module_sp.get() == this)
    {
        SymbolVendor *symbols = GetSymbolVendor ();
        if (symbols)
            return symbols->FindTypes(sc, name, namespace_decl, append, max_matches, types);
    }
    return 0;
}

uint32_t
Module::FindTypesInNamespace (const SymbolContext& sc,
                              const ConstString &type_name,
                              const ClangNamespaceDecl *namespace_decl,
                              uint32_t max_matches,
                              TypeList& type_list)
{
    const bool append = true;
    return FindTypes_Impl(sc, type_name, namespace_decl, append, max_matches, type_list);
}

uint32_t
Module::FindTypes (const SymbolContext& sc,
                   const ConstString &name,
                   bool exact_match,
                   uint32_t max_matches,
                   TypeList& types)
{
    uint32_t num_matches = 0;
    const char *type_name_cstr = name.GetCString();
    std::string type_scope;
    std::string type_basename;
    const bool append = true;
    TypeClass type_class = eTypeClassAny;
    if (Type::GetTypeScopeAndBasename (type_name_cstr, type_scope, type_basename, type_class))
    {
        // Check if "name" starts with "::" which means the qualified type starts
        // from the root namespace and implies and exact match. The typenames we
        // get back from clang do not start with "::" so we need to strip this off
        // in order to get the qualfied names to match

        if (type_scope.size() >= 2 && type_scope[0] == ':' && type_scope[1] == ':')
        {
            type_scope.erase(0,2);
            exact_match = true;
        }
        ConstString type_basename_const_str (type_basename.c_str());
        if (FindTypes_Impl(sc, type_basename_const_str, NULL, append, max_matches, types))
        {
            types.RemoveMismatchedTypes (type_scope, type_basename, type_class, exact_match);
            num_matches = types.GetSize();
        }
    }
    else
    {
        // The type is not in a namespace/class scope, just search for it by basename
        if (type_class != eTypeClassAny)
        {
            // The "type_name_cstr" will have been modified if we have a valid type class
            // prefix (like "struct", "class", "union", "typedef" etc).
            num_matches = FindTypes_Impl(sc, ConstString(type_name_cstr), NULL, append, max_matches, types);
            types.RemoveMismatchedTypes (type_class);
            num_matches = types.GetSize();
        }
        else
        {
            num_matches = FindTypes_Impl(sc, name, NULL, append, max_matches, types);
        }
    }
    
    return num_matches;
    
}

//uint32_t
//Module::FindTypes(const SymbolContext& sc, const RegularExpression& regex, bool append, uint32_t max_matches, Type::Encoding encoding, const char *udt_name, TypeList& types)
//{
//  Timer scoped_timer(__PRETTY_FUNCTION__);
//  SymbolVendor *symbols = GetSymbolVendor ();
//  if (symbols)
//      return symbols->FindTypes(sc, regex, append, max_matches, encoding, udt_name, types);
//  return 0;
//
//}

SymbolVendor*
Module::GetSymbolVendor (bool can_create)
{
    Mutex::Locker locker (m_mutex);
    if (m_did_load_symbol_vendor == false && can_create)
    {
        ObjectFile *obj_file = GetObjectFile ();
        if (obj_file != NULL)
        {
            Timer scoped_timer(__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
            m_symfile_ap.reset(SymbolVendor::FindPlugin(shared_from_this()));
            m_did_load_symbol_vendor = true;
        }
    }
    return m_symfile_ap.get();
}

void
Module::SetFileSpecAndObjectName (const FileSpec &file, const ConstString &object_name)
{
    // Container objects whose paths do not specify a file directly can call
    // this function to correct the file and object names.
    m_file = file;
    m_mod_time = file.GetModificationTime();
    m_object_name = object_name;
}

const ArchSpec&
Module::GetArchitecture () const
{
    return m_arch;
}

void
Module::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    Mutex::Locker locker (m_mutex);

    if (level >= eDescriptionLevelFull)
    {
        if (m_arch.IsValid())
            s->Printf("(%s) ", m_arch.GetArchitectureName());
    }

    if (level == eDescriptionLevelBrief)
    {
        const char *filename = m_file.GetFilename().GetCString();
        if (filename)
            s->PutCString (filename);
    }
    else
    {
        char path[PATH_MAX];
        if (m_file.GetPath(path, sizeof(path)))
            s->PutCString(path);
    }

    const char *object_name = m_object_name.GetCString();
    if (object_name)
        s->Printf("(%s)", object_name);
}

void
Module::ReportError (const char *format, ...)
{
    if (format && format[0])
    {
        StreamString strm;
        strm.PutCString("error: ");
        GetDescription(&strm, lldb::eDescriptionLevelBrief);
        strm.PutChar (' ');
        va_list args;
        va_start (args, format);
        strm.PrintfVarArg(format, args);
        va_end (args);
        
        const int format_len = strlen(format);
        if (format_len > 0)
        {
            const char last_char = format[format_len-1];
            if (last_char != '\n' || last_char != '\r')
                strm.EOL();
        }
        Host::SystemLog (Host::eSystemLogError, "%s", strm.GetString().c_str());

    }
}

bool
Module::FileHasChanged () const
{
    if (m_file_has_changed == false)
        m_file_has_changed = (m_file.GetModificationTime() != m_mod_time);
    return m_file_has_changed;
}

void
Module::ReportErrorIfModifyDetected (const char *format, ...)
{
    if (m_first_file_changed_log == false)
    {
        if (FileHasChanged ())
        {
            m_first_file_changed_log = true;
            if (format)
            {
                StreamString strm;
                strm.PutCString("error: the object file ");
                GetDescription(&strm, lldb::eDescriptionLevelFull);
                strm.PutCString (" has been modified\n");
                
                va_list args;
                va_start (args, format);
                strm.PrintfVarArg(format, args);
                va_end (args);
                
                const int format_len = strlen(format);
                if (format_len > 0)
                {
                    const char last_char = format[format_len-1];
                    if (last_char != '\n' || last_char != '\r')
                        strm.EOL();
                }
                strm.PutCString("The debug session should be aborted as the original debug information has been overwritten.\n");
                Host::SystemLog (Host::eSystemLogError, "%s", strm.GetString().c_str());
            }
        }
    }
}

void
Module::ReportWarning (const char *format, ...)
{
    if (format && format[0])
    {
        StreamString strm;
        strm.PutCString("warning: ");
        GetDescription(&strm, lldb::eDescriptionLevelFull);
        strm.PutChar (' ');
        
        va_list args;
        va_start (args, format);
        strm.PrintfVarArg(format, args);
        va_end (args);
        
        const int format_len = strlen(format);
        if (format_len > 0)
        {
            const char last_char = format[format_len-1];
            if (last_char != '\n' || last_char != '\r')
                strm.EOL();
        }
        Host::SystemLog (Host::eSystemLogWarning, "%s", strm.GetString().c_str());        
    }
}

void
Module::LogMessage (Log *log, const char *format, ...)
{
    if (log)
    {
        StreamString log_message;
        GetDescription(&log_message, lldb::eDescriptionLevelFull);
        log_message.PutCString (": ");
        va_list args;
        va_start (args, format);
        log_message.PrintfVarArg (format, args);
        va_end (args);
        log->PutCString(log_message.GetString().c_str());
    }
}

void
Module::LogMessageVerboseBacktrace (Log *log, const char *format, ...)
{
    if (log)
    {
        StreamString log_message;
        GetDescription(&log_message, lldb::eDescriptionLevelFull);
        log_message.PutCString (": ");
        va_list args;
        va_start (args, format);
        log_message.PrintfVarArg (format, args);
        va_end (args);
        if (log->GetVerbose())
            Host::Backtrace (log_message, 1024);
        log->PutCString(log_message.GetString().c_str());
    }
}

void
Module::Dump(Stream *s)
{
    Mutex::Locker locker (m_mutex);
    //s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    s->Printf("Module %s/%s%s%s%s\n",
              m_file.GetDirectory().AsCString(),
              m_file.GetFilename().AsCString(),
              m_object_name ? "(" : "",
              m_object_name ? m_object_name.GetCString() : "",
              m_object_name ? ")" : "");

    s->IndentMore();
    ObjectFile *objfile = GetObjectFile ();

    if (objfile)
        objfile->Dump(s);

    SymbolVendor *symbols = GetSymbolVendor ();

    if (symbols)
        symbols->Dump(s);

    s->IndentLess();
}


TypeList*
Module::GetTypeList ()
{
    SymbolVendor *symbols = GetSymbolVendor ();
    if (symbols)
        return &symbols->GetTypeList();
    return NULL;
}

const ConstString &
Module::GetObjectName() const
{
    return m_object_name;
}

ObjectFile *
Module::GetObjectFile()
{
    Mutex::Locker locker (m_mutex);
    if (m_did_load_objfile == false)
    {
        m_did_load_objfile = true;
        Timer scoped_timer(__PRETTY_FUNCTION__,
                           "Module::GetObjectFile () module = %s", GetFileSpec().GetFilename().AsCString(""));
        DataBufferSP file_data_sp;
        m_objfile_sp = ObjectFile::FindPlugin (shared_from_this(), 
                                               &m_file, 
                                               m_object_offset, 
                                               m_file.GetByteSize(), 
                                               file_data_sp);
        if (m_objfile_sp)
        {
			// Once we get the object file, update our module with the object file's 
			// architecture since it might differ in vendor/os if some parts were
			// unknown.
            m_objfile_sp->GetArchitecture (m_arch);
        }
    }
    return m_objfile_sp.get();
}


const Symbol *
Module::FindFirstSymbolWithNameAndType (const ConstString &name, SymbolType symbol_type)
{
    Timer scoped_timer(__PRETTY_FUNCTION__,
                       "Module::FindFirstSymbolWithNameAndType (name = %s, type = %i)",
                       name.AsCString(),
                       symbol_type);
    ObjectFile *objfile = GetObjectFile();
    if (objfile)
    {
        Symtab *symtab = objfile->GetSymtab();
        if (symtab)
            return symtab->FindFirstSymbolWithNameAndType (name, symbol_type, Symtab::eDebugAny, Symtab::eVisibilityAny);
    }
    return NULL;
}
void
Module::SymbolIndicesToSymbolContextList (Symtab *symtab, std::vector<uint32_t> &symbol_indexes, SymbolContextList &sc_list)
{
    // No need to protect this call using m_mutex all other method calls are
    // already thread safe.

    size_t num_indices = symbol_indexes.size();
    if (num_indices > 0)
    {
        SymbolContext sc;
        CalculateSymbolContext (&sc);
        for (size_t i = 0; i < num_indices; i++)
        {
            sc.symbol = symtab->SymbolAtIndex (symbol_indexes[i]);
            if (sc.symbol)
                sc_list.Append (sc);
        }
    }
}

size_t
Module::FindSymbolsWithNameAndType (const ConstString &name, SymbolType symbol_type, SymbolContextList &sc_list)
{
    // No need to protect this call using m_mutex all other method calls are
    // already thread safe.


    Timer scoped_timer(__PRETTY_FUNCTION__,
                       "Module::FindSymbolsWithNameAndType (name = %s, type = %i)",
                       name.AsCString(),
                       symbol_type);
    const size_t initial_size = sc_list.GetSize();
    ObjectFile *objfile = GetObjectFile ();
    if (objfile)
    {
        Symtab *symtab = objfile->GetSymtab();
        if (symtab)
        {
            std::vector<uint32_t> symbol_indexes;
            symtab->FindAllSymbolsWithNameAndType (name, symbol_type, symbol_indexes);
            SymbolIndicesToSymbolContextList (symtab, symbol_indexes, sc_list);
        }
    }
    return sc_list.GetSize() - initial_size;
}

size_t
Module::FindSymbolsMatchingRegExAndType (const RegularExpression &regex, SymbolType symbol_type, SymbolContextList &sc_list)
{
    // No need to protect this call using m_mutex all other method calls are
    // already thread safe.

    Timer scoped_timer(__PRETTY_FUNCTION__,
                       "Module::FindSymbolsMatchingRegExAndType (regex = %s, type = %i)",
                       regex.GetText(),
                       symbol_type);
    const size_t initial_size = sc_list.GetSize();
    ObjectFile *objfile = GetObjectFile ();
    if (objfile)
    {
        Symtab *symtab = objfile->GetSymtab();
        if (symtab)
        {
            std::vector<uint32_t> symbol_indexes;
            symtab->FindAllSymbolsMatchingRexExAndType (regex, symbol_type, Symtab::eDebugAny, Symtab::eVisibilityAny, symbol_indexes);
            SymbolIndicesToSymbolContextList (symtab, symbol_indexes, sc_list);
        }
    }
    return sc_list.GetSize() - initial_size;
}

const TimeValue &
Module::GetModificationTime () const
{
    return m_mod_time;
}

bool
Module::IsExecutable ()
{
    if (GetObjectFile() == NULL)
        return false;
    else
        return GetObjectFile()->IsExecutable();
}

bool
Module::IsLoadedInTarget (Target *target)
{
    ObjectFile *obj_file = GetObjectFile();
    if (obj_file)
    {
        SectionList *sections = obj_file->GetSectionList();
        if (sections != NULL)
        {
            size_t num_sections = sections->GetSize();
            for (size_t sect_idx = 0; sect_idx < num_sections; sect_idx++)
            {
                SectionSP section_sp = sections->GetSectionAtIndex(sect_idx);
                if (section_sp->GetLoadBaseAddress(target) != LLDB_INVALID_ADDRESS)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

bool
Module::LoadScriptingResourceInTarget (Target *target, Error& error)
{
    if (!target)
    {
        error.SetErrorString("invalid destination Target");
        return false;
    }
    
    PlatformSP platform_sp(target->GetPlatform());
    
    if (!platform_sp)
    {
        error.SetErrorString("invalid Platform");
        return false;
    }

    ModuleSpec module_spec(GetFileSpec());
    FileSpec scripting_fspec = platform_sp->LocateExecutableScriptingResource(module_spec);
    Debugger &debugger(target->GetDebugger());
    if (scripting_fspec && scripting_fspec.Exists())
    {
        ScriptInterpreter *script_interpreter = debugger.GetCommandInterpreter().GetScriptInterpreter();
        if (script_interpreter)
        {
            StreamString scripting_stream;
            scripting_fspec.Dump(&scripting_stream);
            bool did_load = script_interpreter->LoadScriptingModule(scripting_stream.GetData(), false, true, error);
            if (!did_load)
                return false;
        }
        else
        {
            error.SetErrorString("invalid ScriptInterpreter");
            return false;
        }
    }
    return true;
}

bool
Module::SetArchitecture (const ArchSpec &new_arch)
{
    if (!m_arch.IsValid())
    {
        m_arch = new_arch;
        return true;
    }    
    return m_arch == new_arch;
}

bool 
Module::SetLoadAddress (Target &target, lldb::addr_t offset, bool &changed)
{
    size_t num_loaded_sections = 0;
    ObjectFile *objfile = GetObjectFile();
    if (objfile)
    {
        SectionList *section_list = objfile->GetSectionList ();
        if (section_list)
        {
            const size_t num_sections = section_list->GetSize();
            size_t sect_idx = 0;
            for (sect_idx = 0; sect_idx < num_sections; ++sect_idx)
            {
                // Iterate through the object file sections to find the
                // first section that starts of file offset zero and that
                // has bytes in the file...
                SectionSP section_sp (section_list->GetSectionAtIndex (sect_idx));
                // Only load non-thread specific sections when given a slide
                if (section_sp && !section_sp->IsThreadSpecific())
                {
                    if (target.GetSectionLoadList().SetSectionLoadAddress (section_sp, section_sp->GetFileAddress() + offset))
                        ++num_loaded_sections;
                }
            }
        }
    }
    changed = num_loaded_sections > 0;
    return num_loaded_sections > 0;
}


bool
Module::MatchesModuleSpec (const ModuleSpec &module_ref)
{
    const UUID &uuid = module_ref.GetUUID();
    
    if (uuid.IsValid())
    {
        // If the UUID matches, then nothing more needs to match...
        if (uuid == GetUUID())
            return true;
        else
            return false;
    }
    
    const FileSpec &file_spec = module_ref.GetFileSpec();
    if (file_spec)
    {
        if (!FileSpec::Equal (file_spec, m_file, file_spec.GetDirectory()))
            return false;
    }

    const FileSpec &platform_file_spec = module_ref.GetPlatformFileSpec();
    if (platform_file_spec)
    {
        if (!FileSpec::Equal (platform_file_spec, GetPlatformFileSpec (), platform_file_spec.GetDirectory()))
            return false;
    }
    
    const ArchSpec &arch = module_ref.GetArchitecture();
    if (arch.IsValid())
    {
        if (m_arch != arch)
            return false;
    }
    
    const ConstString &object_name = module_ref.GetObjectName();
    if (object_name)
    {
        if (object_name != GetObjectName())
            return false;
    }
    return true;
}

bool
Module::FindSourceFile (const FileSpec &orig_spec, FileSpec &new_spec) const
{
    Mutex::Locker locker (m_mutex);
    return m_source_mappings.FindFile (orig_spec, new_spec);
}

bool
Module::RemapSourceFile (const char *path, std::string &new_path) const
{
    Mutex::Locker locker (m_mutex);
    return m_source_mappings.RemapPath(path, new_path);
}

uint32_t
Module::GetVersion (uint32_t *versions, uint32_t num_versions)
{
    ObjectFile *obj_file = GetObjectFile();
    if (obj_file)
        return obj_file->GetVersion (versions, num_versions);
        
    if (versions && num_versions)
    {
        for (uint32_t i=0; i<num_versions; ++i)
            versions[i] = UINT32_MAX;
    }
    return 0;
}
