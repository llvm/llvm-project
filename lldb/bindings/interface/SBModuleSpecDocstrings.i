%feature("docstring", "
    Get const accessor for the module file.

    This function returns the file for the module on the host system
    that is running LLDB. This can differ from the path on the
    platform since we might be doing remote debugging.

    @return
        A const reference to the file specification object."
) lldb::SBModuleSpec::GetFileSpec;

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
        A const reference to the file specification object."
) lldb::SBModuleSpec::GetPlatformFileSpec;

%feature("docstring",
"Represents a list of :py:class:`SBModuleSpec`."
) lldb::SBModuleSpecList;
