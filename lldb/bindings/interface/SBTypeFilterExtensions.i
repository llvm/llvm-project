STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeFilter, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeFilter {
#ifdef SWIGPYTHON
        %pythoncode %{
            options = property(GetOptions, SetOptions)
            count = property(GetNumberOfExpressionPaths)
        %}
#endif
}
