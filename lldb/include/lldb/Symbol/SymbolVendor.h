//===-- SymbolVendor.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolVendor_h_
#define liblldb_SymbolVendor_h_

#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/Core/ModuleChild.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/TypeList.h"


namespace lldb_private {

//----------------------------------------------------------------------
// The symbol vendor class is designed to abstract the process of
// searching for debug information for a given module. Platforms can
// subclass this class and provide extra ways to find debug information.
// Examples would be a subclass that would allow for locating a stand
// alone debug file, parsing debug maps, or runtime data in the object
// files. A symbol vendor can use multiple sources (SymbolFile
// objects) to provide the information and only parse as deep as needed
// in order to provide the information that is requested.
//----------------------------------------------------------------------
class SymbolVendor :
    public ModuleChild,
    public PluginInterface
{
public:
    static SymbolVendor*
    FindPlugin (const lldb::ModuleSP &module_sp,
                lldb_private::Stream *feedback_strm);

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    SymbolVendor(const lldb::ModuleSP &module_sp);

    virtual
    ~SymbolVendor();

    void
    AddSymbolFileRepresentation(const lldb::ObjectFileSP &objfile_sp);

    virtual void
    Dump(Stream *s);

    virtual lldb::LanguageType
    ParseCompileUnitLanguage (const SymbolContext& sc);
    
    virtual size_t
    ParseCompileUnitFunctions (const SymbolContext& sc);

    virtual bool
    ParseCompileUnitLineTable (const SymbolContext& sc);

    virtual bool
    ParseCompileUnitSupportFiles (const SymbolContext& sc,
                                  FileSpecList& support_files);

    virtual size_t
    ParseFunctionBlocks (const SymbolContext& sc);

    virtual size_t
    ParseTypes (const SymbolContext& sc);

    virtual size_t
    ParseVariablesForContext (const SymbolContext& sc);

    virtual Type*
    ResolveTypeUID(lldb::user_id_t type_uid);

    virtual uint32_t
    ResolveSymbolContext (const Address& so_addr,
                          uint32_t resolve_scope,
                          SymbolContext& sc);

    virtual uint32_t
    ResolveSymbolContext (const FileSpec& file_spec,
                          uint32_t line,
                          bool check_inlines,
                          uint32_t resolve_scope,
                          SymbolContextList& sc_list);

    virtual size_t
    FindGlobalVariables (const ConstString &name,
                         const ClangNamespaceDecl *namespace_decl,
                         bool append,
                         size_t max_matches,
                         VariableList& variables);

    virtual size_t
    FindGlobalVariables (const RegularExpression& regex,
                         bool append,
                         size_t max_matches,
                         VariableList& variables);

    virtual size_t
    FindFunctions (const ConstString &name,
                   const ClangNamespaceDecl *namespace_decl,
                   uint32_t name_type_mask,
                   bool include_inlines,
                   bool append,
                   SymbolContextList& sc_list);

    virtual size_t
    FindFunctions (const RegularExpression& regex,
                   bool include_inlines,
                   bool append,
                   SymbolContextList& sc_list);

    virtual size_t
    FindTypes (const SymbolContext& sc, 
               const ConstString &name,
               const ClangNamespaceDecl *namespace_decl, 
               bool append, 
               size_t max_matches,
               TypeList& types);

    virtual lldb_private::ClangNamespaceDecl
    FindNamespace (const SymbolContext& sc, 
                   const ConstString &name,
                   const ClangNamespaceDecl *parent_namespace_decl);
    
    virtual size_t
    GetNumCompileUnits();

    virtual bool
    SetCompileUnitAtIndex (size_t cu_idx,
                           const lldb::CompUnitSP &cu_sp);

    virtual lldb::CompUnitSP
    GetCompileUnitAtIndex(size_t idx);

    TypeList&
    GetTypeList()
    {
        return m_type_list;
    }

    const TypeList&
    GetTypeList() const
    {
        return m_type_list;
    }

    SymbolFile *
    GetSymbolFile()
    {
        return m_sym_file_ap.get();
    }

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

protected:
    //------------------------------------------------------------------
    // Classes that inherit from SymbolVendor can see and modify these
    //------------------------------------------------------------------
    typedef std::vector<lldb::CompUnitSP> CompileUnits;
    typedef CompileUnits::iterator CompileUnitIter;
    typedef CompileUnits::const_iterator CompileUnitConstIter;

    TypeList m_type_list; // Uniqued types for all parsers owned by this module
    CompileUnits m_compile_units; // The current compile units
    lldb::ObjectFileSP m_objfile_sp;    // Keep a reference to the object file in case it isn't the same as the module object file (debug symbols in a separate file)
    std::unique_ptr<SymbolFile> m_sym_file_ap; // A single symbol file. Suclasses can add more of these if needed.

private:
    //------------------------------------------------------------------
    // For SymbolVendor only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (SymbolVendor);
};


} // namespace lldb_private

#endif  // liblldb_SymbolVendor_h_
