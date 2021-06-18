//===-- SWIG Interface for SBProcessInfo-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Describes an existing process and any discoverable information that pertains to
that process."
) SBProcessInfo;

class SBProcessInfo
{
public:
    SBProcessInfo();

    SBProcessInfo (const SBProcessInfo &rhs);

    ~SBProcessInfo ();

    bool
    IsValid ();

    explicit operator bool() const;

    const char *
    GetName ();

    SBFileSpec
    GetExecutableFile ();

    lldb::pid_t
    GetProcessID ();

    uint32_t
    GetUserID ();

    uint32_t
    GetGroupID ();

    bool
    UserIDIsValid ();

    bool
    GroupIDIsValid ();

    uint32_t
    GetEffectiveUserID ();

    uint32_t
    GetEffectiveGroupID ();

    bool
    EffectiveUserIDIsValid ();

    bool
    EffectiveGroupIDIsValid ();

    lldb::pid_t
    GetParentProcessID ();

    %feature("docstring",
    "Return the target triple (arch-vendor-os) for the described process."
    ) GetTriple;
    const char *
    GetTriple ();

    %feature("docstring",
    "Return the number of arguments given to the described process."
    ) GetNumArguments;
    uint32_t
    GetNumArguments ();

    %feature("autodoc", "
    GetArgumentAtIndex(int index) -> string
    Return the specified argument given to the described process."
    ) GetArgumentAtIndex;
    const char *
    GetArgumentAtIndex (uint32_t index);

    %feature("docstring",
    "Return the environment variables for the described process."
    ) GetEnvironment;
    SBEnvironment
    GetEnvironment ();
};

} // namespace lldb
