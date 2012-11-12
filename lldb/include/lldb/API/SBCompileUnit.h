//===-- SBCompileUnit.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBCompileUnit_h_
#define LLDB_SBCompileUnit_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBFileSpec.h"

namespace lldb {

class SBCompileUnit
{
public:

    SBCompileUnit ();

    SBCompileUnit (const lldb::SBCompileUnit &rhs);

    ~SBCompileUnit ();

    const lldb::SBCompileUnit &
    operator = (const lldb::SBCompileUnit &rhs);

    bool
    IsValid () const;

    lldb::SBFileSpec
    GetFileSpec () const;

    uint32_t
    GetNumLineEntries () const;

    lldb::SBLineEntry
    GetLineEntryAtIndex (uint32_t idx) const;

    uint32_t
    FindLineEntryIndex (uint32_t start_idx,
                        uint32_t line,
                        lldb::SBFileSpec *inline_file_spec) const;

    uint32_t
    FindLineEntryIndex (uint32_t start_idx,
                        uint32_t line,
                        lldb::SBFileSpec *inline_file_spec,
                        bool exact) const;

    SBFileSpec
    GetSupportFileAtIndex (uint32_t idx) const;

    uint32_t
    GetNumSupportFiles () const;

    uint32_t
    FindSupportFileIndex (uint32_t start_idx, const SBFileSpec &sb_file, bool full);

    bool
    operator == (const lldb::SBCompileUnit &rhs) const;

    bool
    operator != (const lldb::SBCompileUnit &rhs) const;

    bool
    GetDescription (lldb::SBStream &description);

private:
    friend class SBAddress;
    friend class SBFrame;
    friend class SBSymbolContext;
    friend class SBModule;

    SBCompileUnit (lldb_private::CompileUnit *lldb_object_ptr);

    const lldb_private::CompileUnit *
    operator->() const;

    const lldb_private::CompileUnit &
    operator*() const;
    
    lldb_private::CompileUnit *
    get ();
    
    void
    reset (lldb_private::CompileUnit *lldb_object_ptr);

    lldb_private::CompileUnit *m_opaque_ptr;
};


} // namespace lldb

#endif // LLDB_SBCompileUnit_h_
