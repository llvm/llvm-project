//===-- SBModule.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBModule_h_
#define LLDB_SBModule_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBSection.h"
#include "lldb/API/SBSymbolContext.h"
#include "lldb/API/SBValueList.h"

namespace lldb {

class SBModule
{
public:

    SBModule ();

    SBModule (const SBModule &rhs);

    const SBModule &
    operator = (const SBModule &rhs);

    SBModule (lldb::SBProcess &process, 
              lldb::addr_t header_addr);

    ~SBModule ();

    bool
    IsValid () const;

    void
    Clear();

    //------------------------------------------------------------------
    /// Get const accessor for the module file specification.
    ///
    /// This function returns the file for the module on the host system
    /// that is running LLDB. This can differ from the path on the 
    /// platform since we might be doing remote debugging.
    ///
    /// @return
    ///     A const reference to the file specification object.
    //------------------------------------------------------------------
    lldb::SBFileSpec
    GetFileSpec () const;

    //------------------------------------------------------------------
    /// Get accessor for the module platform file specification.
    ///
    /// Platform file refers to the path of the module as it is known on
    /// the remote system on which it is being debugged. For local 
    /// debugging this is always the same as Module::GetFileSpec(). But
    /// remote debugging might mention a file '/usr/lib/liba.dylib'
    /// which might be locally downloaded and cached. In this case the
    /// platform file could be something like:
    /// '/tmp/lldb/platform-cache/remote.host.computer/usr/lib/liba.dylib'
    /// The file could also be cached in a local developer kit directory.
    ///
    /// @return
    ///     A const reference to the file specification object.
    //------------------------------------------------------------------
    lldb::SBFileSpec
    GetPlatformFileSpec () const;

    bool
    SetPlatformFileSpec (const lldb::SBFileSpec &platform_file);

    lldb::ByteOrder
    GetByteOrder ();
    
    uint32_t
    GetAddressByteSize();
    
    const char *
    GetTriple ();

    const uint8_t *
    GetUUIDBytes () const;

    const char *
    GetUUIDString () const;

    bool
    operator == (const lldb::SBModule &rhs) const;

    bool
    operator != (const lldb::SBModule &rhs) const;

    lldb::SBSection
    FindSection (const char *sect_name);

    lldb::SBAddress
    ResolveFileAddress (lldb::addr_t vm_addr);

    lldb::SBSymbolContext
    ResolveSymbolContextForAddress (const lldb::SBAddress& addr, 
                                    uint32_t resolve_scope);

    bool
    GetDescription (lldb::SBStream &description);

    uint32_t
    GetNumCompileUnits();

    lldb::SBCompileUnit
    GetCompileUnitAtIndex (uint32_t);

    size_t
    GetNumSymbols ();
    
    lldb::SBSymbol
    GetSymbolAtIndex (size_t idx);

    lldb::SBSymbol
    FindSymbol (const char *name,
                lldb::SymbolType type = eSymbolTypeAny);

    lldb::SBSymbolContextList
    FindSymbols (const char *name,
                 lldb::SymbolType type = eSymbolTypeAny);

    size_t
    GetNumSections ();
    
    lldb::SBSection
    GetSectionAtIndex (size_t idx);
    //------------------------------------------------------------------
    /// Find functions by name.
    ///
    /// @param[in] name
    ///     The name of the function we are looking for.
    ///
    /// @param[in] name_type_mask
    ///     A logical OR of one or more FunctionNameType enum bits that
    ///     indicate what kind of names should be used when doing the
    ///     lookup. Bits include fully qualified names, base names,
    ///     C++ methods, or ObjC selectors. 
    ///     See FunctionNameType for more details.
    ///
    /// @return
    ///     A lldb::SBSymbolContextList that gets filled in with all of 
    ///     the symbol contexts for all the matches.
    //------------------------------------------------------------------
    lldb::SBSymbolContextList
    FindFunctions (const char *name, 
                   uint32_t name_type_mask = lldb::eFunctionNameTypeAny);

    //------------------------------------------------------------------
    /// Find global and static variables by name.
    ///
    /// @param[in] target
    ///     A valid SBTarget instance representing the debuggee.
    ///
    /// @param[in] name
    ///     The name of the global or static variable we are looking
    ///     for.
    ///
    /// @param[in] max_matches
    ///     Allow the number of matches to be limited to \a max_matches.
    ///
    /// @return
    ///     A list of matched variables in an SBValueList.
    //------------------------------------------------------------------
    lldb::SBValueList
    FindGlobalVariables (lldb::SBTarget &target, 
                         const char *name, 
                         uint32_t max_matches);
    
    //------------------------------------------------------------------
    /// Find the first global (or static) variable by name.
    ///
    /// @param[in] target
    ///     A valid SBTarget instance representing the debuggee.
    ///
    /// @param[in] name
    ///     The name of the global or static variable we are looking
    ///     for.
    ///
    /// @return
    ///     An SBValue that gets filled in with the found variable (if any).
    //------------------------------------------------------------------
    lldb::SBValue
    FindFirstGlobalVariable (lldb::SBTarget &target, const char *name);
    
    lldb::SBType
    FindFirstType (const char* name);
    
    lldb::SBTypeList
    FindTypes (const char* type);
    
    lldb::SBType
    GetBasicType(lldb::BasicType type);
    
    //------------------------------------------------------------------
    /// Get the module version numbers.
    ///
    /// Many object files have a set of version numbers that describe
    /// the version of the executable or shared library. Typically there
    /// are major, minor and build, but there may be more. This function
    /// will extract the versions from object files if they are available.
    ///
    /// If \a versions is NULL, or if \a num_versions is 0, the return
    /// value will indicate how many version numbers are available in
    /// this object file. Then a subsequent call can be made to this 
    /// function with a value of \a versions and \a num_versions that
    /// has enough storage to store some or all version numbers.
    ///
    /// @param[out] versions
    ///     A pointer to an array of uint32_t types that is \a num_versions
    ///     long. If this value is NULL, the return value will indicate
    ///     how many version numbers are required for a subsequent call
    ///     to this function so that all versions can be retrieved. If
    ///     the value is non-NULL, then at most \a num_versions of the
    ///     existing versions numbers will be filled into \a versions.
    ///     If there is no version information available, \a versions
    ///     will be filled with \a num_versions UINT32_MAX values
    ///     and zero will be returned.
    ///
    /// @param[in] num_versions
    ///     The maximum number of entries to fill into \a versions. If
    ///     this value is zero, then the return value will indicate
    ///     how many version numbers there are in total so another call
    ///     to this function can be make with adequate storage in
    ///     \a versions to get all of the version numbers. If \a
    ///     num_versions is less than the actual number of version 
    ///     numbers in this object file, only \a num_versions will be
    ///     filled into \a versions (if \a versions is non-NULL).
    ///
    /// @return
    ///     This function always returns the number of version numbers
    ///     that this object file has regardless of the number of
    ///     version numbers that were copied into \a versions. 
    //------------------------------------------------------------------
    uint32_t
    GetVersion (uint32_t *versions, 
                uint32_t num_versions);

private:
    friend class SBAddress;
    friend class SBFrame;
    friend class SBSection;
    friend class SBSymbolContext;
    friend class SBTarget;

    explicit SBModule (const lldb::ModuleSP& module_sp);

    ModuleSP
    GetSP () const;
    
    void
    SetSP (const ModuleSP &module_sp);

    lldb::ModuleSP m_opaque_sp;
};


} // namespace lldb

#endif // LLDB_SBModule_h_
