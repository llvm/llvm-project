STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeFormat, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeFormat {
#ifdef SWIGPYTHON
        %pythoncode %{
            def __eq__(self, other):
                return not self.__ne__(other)

            def __int__(self):
                pass

            def __hex__(self):
                pass

            def __oct__(self):
                pass

            def __len__(self):
                pass

            def __iter__(self):
                pass

            format = property(GetFormat, SetFormat)
            options = property(GetOptions, SetOptions)
        %}
#endif
}
