//===-- SWIG Interface for SBProcessInfo-------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
};

} // namespace lldb
