STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeNameSpecifier, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeNameSpecifier {
#ifdef SWIGPYTHON
        %pythoncode %{
            name = property(GetName)
            is_regex = property(IsRegex)
        %}
#endif
}
