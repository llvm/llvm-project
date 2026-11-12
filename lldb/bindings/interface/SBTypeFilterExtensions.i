STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeFilter, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeFilter {
#ifdef SWIGPYTHON
        %pythoncode %{
            # operator== is a free function, which swig does not handle, so we inject
            # our own equality operator here
            def __eq__(self, other):
                return not self.__ne__(other)

            options = property(GetOptions, SetOptions)
            count = property(GetNumberOfExpressionPaths)
        %}
#endif
}
