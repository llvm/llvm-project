//===-- SWIG Interface for SBFileSpecList -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
