//===-- SWIG Interface for SBStringList -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBStringList
{
public:

    SBStringList ();

    SBStringList (const lldb::SBStringList &rhs);
    
    ~SBStringList ();

    bool
    IsValid() const;

    void
    AppendString (const char *str);

    void
    AppendList (const char **strv, int strc);

    void
    AppendList (const lldb::SBStringList &strings);

    uint32_t
    GetSize () const;

    const char *
    GetStringAtIndex (size_t idx);

    void
    Clear ();
};

} // namespace lldb
