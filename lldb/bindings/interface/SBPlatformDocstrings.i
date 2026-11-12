%feature("docstring",
"Describes how :py:class:`SBPlatform.ConnectRemote` connects to a remote platform."
) lldb::SBPlatformConnectOptions;

%feature("docstring",
"Represents a shell command that can be run by :py:class:`SBPlatform.Run`."
) lldb::SBPlatformShellCommand;

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

Every :py:class:`SBTarget` has a corresponding SBPlatform. The platform can be
specified upon target creation, or the currently selected platform
will attempt to be used when creating the target automatically as long
as the currently selected platform matches the target architecture
and executable type. If the architecture or executable type do not match,
a suitable platform will be found automatically."

) lldb::SBPlatform;
