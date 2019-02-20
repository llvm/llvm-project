//===-- SWIG Interface for SBPlatform ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

    
class SBPlatformConnectOptions
{
public:
    SBPlatformConnectOptions (const char *url);
    
    SBPlatformConnectOptions (const SBPlatformConnectOptions &rhs);
    
    ~SBPlatformConnectOptions ();
    
    const char *
    GetURL();
    
    void
    SetURL(const char *url);
    
    bool
    GetRsyncEnabled();
    
    void
    EnableRsync (const char *options,
                 const char *remote_path_prefix,
                 bool omit_remote_hostname);
    
    void
    DisableRsync ();
    
    const char *
    GetLocalCacheDirectory();
    
    void
    SetLocalCacheDirectory(const char *path);
};

class SBPlatformShellCommand
{
public:
    SBPlatformShellCommand (const char *shell_command);
    
    SBPlatformShellCommand (const SBPlatformShellCommand &rhs);
    
    ~SBPlatformShellCommand();
    
    void
    Clear();

    const char *
    GetCommand();

    void
    SetCommand(const char *shell_command);
    
    const char *
    GetWorkingDirectory ();

    void
    SetWorkingDirectory (const char *path);

    uint32_t
    GetTimeoutSeconds ();
    
    void
    SetTimeoutSeconds (uint32_t sec);
    
    int
    GetSignal ();
    
    int
    GetStatus ();
    
    const char *
    GetOutput ();
};

%feature("docstring",
"A class that represents a platform that can represent the current host or a remote host debug platform.

The SBPlatform class represents the current host, or a remote host.
It can be connected to a remote platform in order to provide ways
to remotely launch and attach to processes, upload/download files,
create directories, run remote shell commands, find locally cached
versions of files from the remote system, and much more.
         
SBPlatform objects can be created and then used to connect to a remote
platform which allows the SBPlatform to be used to get a list of the
current processes on the remote host, attach to one of those processes,
install programs on the remote system, attach and launch processes,
and much more.

Every SBTarget has a corresponding SBPlatform. The platform can be
specified upon target creation, or the currently selected platform
will attempt to be used when creating the target automatically as long
as the currently selected platform matches the target architecture
and executable type. If the architecture or executable type do not match,
a suitable platform will be found automatically."
         
) SBPlatform;
class SBPlatform
{
public:

    SBPlatform ();

    SBPlatform (const char *);

    ~SBPlatform();
    
    bool
    IsValid () const;

    void
    Clear ();

    const char *
    GetWorkingDirectory();
    
    bool
    SetWorkingDirectory(const char *);

    const char *
    GetName ();
    
    SBError
    ConnectRemote (lldb::SBPlatformConnectOptions &connect_options);
    
    void
    DisconnectRemote ();

    bool
    IsConnected();
    
    const char *
    GetTriple();
    
    const char *
    GetHostname ();
    
    const char *
    GetOSBuild ();
    
    const char *
    GetOSDescription ();
    
    uint32_t
    GetOSMajorVersion ();
    
    uint32_t
    GetOSMinorVersion ();
    
    uint32_t
    GetOSUpdateVersion ();
    
    lldb::SBError
    Get (lldb::SBFileSpec &src, lldb::SBFileSpec &dst);

    lldb::SBError
    Put (lldb::SBFileSpec &src, lldb::SBFileSpec &dst);
    
    lldb::SBError
    Install (lldb::SBFileSpec &src, lldb::SBFileSpec &dst);
    
    lldb::SBError
    Run (lldb::SBPlatformShellCommand &shell_command);

    lldb::SBError
    Launch (lldb::SBLaunchInfo &launch_info);

    lldb::SBError
    Kill (const lldb::pid_t pid);

    lldb::SBError
    MakeDirectory (const char *path, uint32_t file_permissions = lldb::eFilePermissionsDirectoryDefault);
    
    uint32_t
    GetFilePermissions (const char *path);
    
    lldb::SBError
    SetFilePermissions (const char *path, uint32_t file_permissions);

    lldb::SBUnixSignals
    GetUnixSignals();

};

} // namespace lldb
