STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeFormat, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeFormat {
#ifdef SWIGPYTHON
        %pythoncode %{
            # operator== is a free function, which swig does not handle, so we inject
            # our own equality operator here
            def __eq__(self, other):
                return not self.__ne__(other)

            format = property(GetFormat, SetFormat)
            options = property(GetOptions, SetOptions)
        %}
#endif
}
