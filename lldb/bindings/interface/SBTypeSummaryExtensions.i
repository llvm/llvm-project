STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeSummary, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeSummary {
#ifdef SWIGPYTHON
        %pythoncode %{
            options = property(GetOptions, SetOptions)
            is_summary_string = property(IsSummaryString)
            is_function_name = property(IsFunctionName)
            is_function_name = property(IsFunctionCode)
            summary_data = property(GetData)
        %}
#endif
}
