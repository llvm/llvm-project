STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeSummary, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeSummary {
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

            options = property(GetOptions, SetOptions)
            is_summary_string = property(IsSummaryString)
            is_function_name = property(IsFunctionName)
            is_function_name = property(IsFunctionCode)
            summary_data = property(GetData)
        %}
#endif
}
