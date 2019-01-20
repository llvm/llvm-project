//===-- SWIG Interface for SBAttachInfo--------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBAttachInfo
{
public:
    SBAttachInfo ();

    SBAttachInfo (lldb::pid_t pid);

    SBAttachInfo (const char *path, bool wait_for);

    SBAttachInfo (const char *path, bool wait_for, bool async);

    SBAttachInfo (const lldb::SBAttachInfo &rhs);

    lldb::pid_t
    GetProcessID ();

    void
    SetProcessID (lldb::pid_t pid);

    void
    SetExecutable (const char *path);

    void
    SetExecutable (lldb::SBFileSpec exe_file);

    bool
    GetWaitForLaunch ();

    void
    SetWaitForLaunch (bool b);

    void
    SetWaitForLaunch (bool b, bool async);

    bool
    GetIgnoreExisting ();

    void
    SetIgnoreExisting (bool b);

    uint32_t
    GetResumeCount ();

    void
    SetResumeCount (uint32_t c);

    const char *
    GetProcessPluginName ();

    void
    SetProcessPluginName (const char *plugin_name);

    uint32_t
    GetUserID();

    uint32_t
    GetGroupID();

    bool
    UserIDIsValid ();

    bool
    GroupIDIsValid ();

    void
    SetUserID (uint32_t uid);

    void
    SetGroupID (uint32_t gid);

    uint32_t
    GetEffectiveUserID();

    uint32_t
    GetEffectiveGroupID();

    bool
    EffectiveUserIDIsValid ();

    bool
    EffectiveGroupIDIsValid ();

    void
    SetEffectiveUserID (uint32_t uid);

    void
    SetEffectiveGroupID (uint32_t gid);

    lldb::pid_t
    GetParentProcessID ();

    void
    SetParentProcessID (lldb::pid_t pid);

    bool
    ParentProcessIDIsValid();

    lldb::SBListener
    GetListener ();

    void
    SetListener (lldb::SBListener &listener);
};

} // namespace lldb
