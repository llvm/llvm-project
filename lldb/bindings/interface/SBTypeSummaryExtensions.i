STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeSummary, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeSummary {
#ifdef SWIGPYTHON
        %pythoncode %{
            # operator== is a free function, which swig does not handle, so we inject
            # our own equality operator here
            def __eq__(self, other):
                return not self.__ne__(other)

            options = property(GetOptions, SetOptions)
            is_summary_string = property(IsSummaryString)
            is_function_name = property(IsFunctionName)
            is_function_name = property(IsFunctionCode)
            summary_data = property(GetData)
        %}
#endif
}
