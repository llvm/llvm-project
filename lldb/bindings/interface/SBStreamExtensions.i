%extend lldb::SBStream {
#ifdef SWIGPYTHON
    %pythoncode%{
      def __len__(self):
        return self.GetSize()
    %}
#endif

    %feature("autodoc", "DEPRECATED, use RedirectToFile") RedirectToFileHandle;
    void
    RedirectToFileHandle (lldb::FileSP file, bool transfer_fh_ownership) {
        self->RedirectToFile(file);
    }
}
