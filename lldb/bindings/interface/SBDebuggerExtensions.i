STRING_EXTENSION_OUTSIDE(SBDebugger)

%extend lldb::SBDebugger {
#ifdef SWIGPYTHON
    %pythoncode %{
        def SetOutputFileHandle(self, file, transfer_ownership):
            "DEPRECATED, use SetOutputFile"
            if file is None:
                import sys
                file = sys.stdout
            self.SetOutputFile(SBFile.Create(file, borrow=True))

        def SetInputFileHandle(self, file, transfer_ownership):
            "DEPRECATED, use SetInputFile"
            if file is None:
                import sys
                file = sys.stdin
            self.SetInputFile(SBFile.Create(file, borrow=True))

        def SetErrorFileHandle(self, file, transfer_ownership):
            "DEPRECATED, use SetErrorFile"
            if file is None:
                import sys
                file = sys.stderr
            self.SetErrorFile(SBFile.Create(file, borrow=True))

        def __iter__(self):
            '''Iterate over all targets in a lldb.SBDebugger object.'''
            return lldb_iter(self, 'GetNumTargets', 'GetTargetAtIndex')

        def __len__(self):
            '''Return the number of targets in a lldb.SBDebugger object.'''
            return self.GetNumTargets()
    %}
#endif

    lldb::FileSP GetInputFileHandle() {
        return self->GetInputFile().GetFile();
    }

    lldb::FileSP GetOutputFileHandle() {
        return self->GetOutputFile().GetFile();
    }

    lldb::FileSP GetErrorFileHandle() {
        return self->GetErrorFile().GetFile();
    }
}
