//===-- SWIG Interface for SBLaunchInfo--------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBLaunchInfo
{
public:
    SBLaunchInfo (const char **argv);

    pid_t
    GetProcessID();

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

    lldb::SBFileSpec
    GetExecutableFile ();

    void
    SetExecutableFile (lldb::SBFileSpec exe_file, bool add_as_first_arg);

    lldb::SBListener
    GetListener ();

    void
    SetListener (lldb::SBListener &listener);

    uint32_t
    GetNumArguments ();

    const char *
    GetArgumentAtIndex (uint32_t idx);

    void
    SetArguments (const char **argv, bool append);

    uint32_t
    GetNumEnvironmentEntries ();

    const char *
    GetEnvironmentEntryAtIndex (uint32_t idx);

    void
    SetEnvironmentEntries (const char **envp, bool append);

    void
    Clear ();

    const char *
    GetWorkingDirectory () const;

    void
    SetWorkingDirectory (const char *working_dir);

    uint32_t
    GetLaunchFlags ();

    void
    SetLaunchFlags (uint32_t flags);

    const char *
    GetProcessPluginName ();

    void
    SetProcessPluginName (const char *plugin_name);

    const char *
    GetShell ();

    void
    SetShell (const char * path);
    
    bool
    GetShellExpandArguments ();
    
    void
    SetShellExpandArguments (bool expand);

    uint32_t
    GetResumeCount ();

    void
    SetResumeCount (uint32_t c);

    bool
    AddCloseFileAction (int fd);

    bool
    AddDuplicateFileAction (int fd, int dup_fd);

    bool
    AddOpenFileAction (int fd, const char *path, bool read, bool write);

    bool
    AddSuppressFileAction (int fd, bool read, bool write);

    void
    SetLaunchEventData (const char *data);

    const char *
    GetLaunchEventData () const;

    bool
    GetDetachOnError() const;

    void
    SetDetachOnError(bool enable);
};

} // namespace lldb
