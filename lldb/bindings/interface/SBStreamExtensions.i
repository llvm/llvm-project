%extend lldb::SBStream {
#ifdef SWIGPYTHON
    // operator== is a free function, which swig does not handle, so we inject
    // our own equality operator here
    %pythoncode%{
	    def __eq__(self, other):
	      return not self.__ne__(other)

	    def __int__(self):
	      pass

	    def __hex__(self):
	      pass

	    def __oct__(self):
	      pass

	    def __len__(self):
	      return self.GetSize()

	    def __iter__(self):
	      pass
    %}
#endif

    %feature("autodoc", "DEPRECATED, use RedirectToFile") RedirectToFileHandle;
    void
    RedirectToFileHandle (lldb::FileSP file, bool transfer_fh_ownership) {
        self->RedirectToFile(file);
    }
}
