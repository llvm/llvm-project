//===-- SWIG Interface for SBEnvironment-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents the environment of a certain process.

Example: ::

  for entry in lldb.debugger.GetSelectedTarget().GetEnvironment().GetEntries():
    print(entry)

") SBEnvironment;
class SBEnvironment {
public:
    SBEnvironment ();

    SBEnvironment (const lldb::SBEnvironment &rhs);

    ~SBEnvironment();

    size_t GetNumValues();

    const char *Get(const char *name);

    const char *GetNameAtIndex(size_t index);

    const char *GetValueAtIndex(size_t index);

    SBStringList GetEntries();

    void PutEntry(const char *name_and_value);

    void SetEntries(const SBStringList &entries, bool append);

    bool Set(const char *name, const char *value, bool overwrite);

    bool Unset(const char *name);

    void Clear();
};

} // namespace lldb
