STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeNameSpecifier, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeNameSpecifier {
#ifdef SWIGPYTHON
        %pythoncode %{
            # operator== is a free function, which swig does not handle, so we inject
            # our own equality operator here
            def __eq__(self, other):
                return not self.__ne__(other)

            name = property(GetName)
            is_regex = property(IsRegex)
        %}
#endif
}
