//===-- SWIG Interface for SBFileSpecList -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBFileSpecList
{
public:
    SBFileSpecList ();

    SBFileSpecList (const lldb::SBFileSpecList &rhs);

    ~SBFileSpecList ();

    uint32_t
    GetSize () const;

    bool
    GetDescription (SBStream &description) const;
    
    void
    Append (const SBFileSpec &sb_file);
    
    bool
    AppendIfUnique (const SBFileSpec &sb_file);
    
    void
    Clear();
    
    uint32_t
    FindFileIndex (uint32_t idx, const SBFileSpec &sb_file, bool full);
    
    const SBFileSpec
    GetFileSpecAtIndex (uint32_t idx) const;

};


} // namespace lldb
