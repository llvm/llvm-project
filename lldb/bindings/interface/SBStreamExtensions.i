%extend lldb::SBStream {
    %feature("autodoc", "DEPRECATED, use RedirectToFile") RedirectToFileHandle;
    void
    RedirectToFileHandle (lldb::FileSP file, bool transfer_fh_ownership) {
        self->RedirectToFile(file);
    }
}
