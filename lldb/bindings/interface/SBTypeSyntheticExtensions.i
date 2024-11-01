STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeSynthetic, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeSynthetic {
#ifdef SWIGPYTHON
        %pythoncode %{
            options = property(GetOptions, SetOptions)
            contains_code = property(IsClassCode, None)
            synthetic_data = property(GetData, None)
        %}
#endif
}
