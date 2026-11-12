STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeSynthetic, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeSynthetic {
#ifdef SWIGPYTHON
        %pythoncode %{
            # operator== is a free function, which swig does not handle, so we inject
            # our own equality operator here
            def __eq__(self, other):
                return not self.__ne__(other)

            options = property(GetOptions, SetOptions)
            contains_code = property(IsClassCode, None)
            synthetic_data = property(GetData, None)
        %}
#endif
}
