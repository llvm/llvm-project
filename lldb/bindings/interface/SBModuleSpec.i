//===-- SWIG Interface for SBModule -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBModuleSpec
{
public:

    SBModuleSpec ();

    SBModuleSpec (const lldb::SBModuleSpec &rhs);

    ~SBModuleSpec ();

    bool
    IsValid () const;

    explicit operator bool() const;

    void
    Clear();

    %feature("docstring", "
    Get const accessor for the module file.

    This function returns the file for the module on the host system
    that is running LLDB. This can differ from the path on the
    platform since we might be doing remote debugging.

    @return
        A const reference to the file specification object.") GetFileSpec;
    lldb::SBFileSpec
    GetFileSpec ();

    void
    SetFileSpec (const lldb::SBFileSpec &fspec);

    %feature("docstring", "
    Get accessor for the module platform file.

    Platform file refers to the path of the module as it is known on
    the remote system on which it is being debugged. For local
    debugging this is always the same as Module::GetFileSpec(). But
    remote debugging might mention a file '/usr/lib/liba.dylib'
    which might be locally downloaded and cached. In this case the
    platform file could be something like:
    '/tmp/lldb/platform-cache/remote.host.computer/usr/lib/liba.dylib'
    The file could also be cached in a local developer kit directory.

    @return
        A const reference to the file specification object.") GetPlatformFileSpec;
    lldb::SBFileSpec
    GetPlatformFileSpec ();

    void
    SetPlatformFileSpec (const lldb::SBFileSpec &fspec);

    lldb::SBFileSpec
    GetSymbolFileSpec ();

    void
    SetSymbolFileSpec (const lldb::SBFileSpec &fspec);

    const char *
    GetObjectName ();

    void
    SetObjectName (const char *name);

    const char *
    GetTriple ();

    void
    SetTriple (const char *triple);

    const uint8_t *
    GetUUIDBytes ();

    size_t
    GetUUIDLength ();

    bool
    SetUUIDBytes (const uint8_t *uuid, size_t uuid_len);

    bool
    GetDescription (lldb::SBStream &description);

    STRING_EXTENSION(SBModuleSpec)
};


class SBModuleSpecList
{
public:
    SBModuleSpecList();

    SBModuleSpecList (const SBModuleSpecList &rhs);

    ~SBModuleSpecList();

    static SBModuleSpecList
    GetModuleSpecifications (const char *path);

    void
    Append (const lldb::SBModuleSpec &spec);

    void
    Append (const lldb::SBModuleSpecList &spec_list);

    lldb::SBModuleSpec
    FindFirstMatchingSpec (const lldb::SBModuleSpec &match_spec);

    lldb::SBModuleSpecList
    FindMatchingSpecs (const lldb::SBModuleSpec &match_spec);

    size_t
    GetSize();

    lldb::SBModuleSpec
    GetSpecAtIndex (size_t i);

    bool
    GetDescription (lldb::SBStream &description);

    STRING_EXTENSION(SBModuleSpecList)
};

} // namespace lldb
