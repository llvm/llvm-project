//===-- SWIG Interface for SBStringList -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

    explicit operator bool() const;

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

    %pythoncode%{
    def __iter__(self):
        '''Iterate over all strings in a lldb.SBStringList object.'''
        return lldb_iter(self, 'GetSize', 'GetStringAtIndex')

    def __len__(self):
        '''Return the number of strings in a lldb.SBStringList object.'''
        return self.GetSize()
    %}
};

} // namespace lldb
