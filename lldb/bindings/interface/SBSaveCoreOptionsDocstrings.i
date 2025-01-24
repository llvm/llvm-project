%feature("docstring",
"A container to specify how to save a core file.

SBSaveCoreOptions includes API's to specify the memory regions and threads to include
when generating a core file. It extends the existing SaveCoreStyle option.

* eSaveCoreFull will save off all thread and memory regions, ignoring the memory regions and threads in
the options object.

* eSaveCoreDirtyOnly pages will capture all threads and all rw- memory regions, in addition to the regions specified
in the options object if they are not already captured.

* eSaveCoreStackOnly will capture all threads, but no memory regions unless specified.

* eSaveCoreCustomOnly Custom defers entirely to the SBSaveCoreOptions object and will only save what is specified. 
  Picking custom and specifying nothing will result in an error being returned.

Note that currently ELF Core files are not supported."
) lldb::SBSaveCoreOptions;

%feature("docstring", "
    Set the plugin name to save a Core file with. Only plugins registered with Plugin manager will be accepted
    Examples are Minidump and Mach-O."
) lldb::SBSaveCoreOptions::SetPluginName;

%feature("docstring", "
    Get the specified plugin name, or None if the name is not set."
) lldb::SBSaveCoreOptions::GetPluginName;

%feature("docstring", "
    Set the lldb.SaveCoreStyle."
) lldb::SBSaveCoreOptions::SetStyle;

%feature("docstring", "
    Get the specified lldb.SaveCoreStyle, or eSaveCoreUnspecified if not set."
) lldb::SBSaveCoreOptions::GetStyle;

%feature("docstring", "
    Set the file path to save the Core file at."
) lldb::SBSaveCoreOptions::SetOutputFile;

%feature("docstring", "
    Get an SBFileSpec corresponding to the specified output path, or none if not set."
) lldb::SBSaveCoreOptions::GetOutputFile;

%feature("docstring", "
    Set the process to save, or unset a process by providing a default SBProcess. 
    Resetting will result in the reset of all process specific options, such as Threads to save."
) lldb::SBSaveCoreOptions::SetProcess;

%feature("docstring", "
    Add an SBThread to be saved, an error will be returned if an SBThread from a different process is specified. 
    The process is set either by the first SBThread added to the options container, or explicitly by the SetProcess call."
) lldb::SBSaveCoreOptions::AddThread;

%feature("docstring", "
    Remove an SBthread if present in the container, returns true if a matching thread was found and removed."
) lldb::SBSaveCoreOptions::RemoveThread;

%feature("docstring", "
    Add a memory region to save, an error will be returned in the region is invalid. 
    Ranges that overlap will be unioned into a single region."
) lldb::SBSaveCoreOptions::AddMemoryRegionToSave;

%feature("docstring", "
    Get an SBThreadCollection of all threads marked to be saved. This collection is not sorted according to insertion order."
) lldb::SBSaveCoreOptions::GetThreadsToSave;

%feature("docstring", "
    Unset all options."
) lldb::SBSaveCoreOptions::Clear;
