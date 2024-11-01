STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeFormat, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeFormat {
#ifdef SWIGPYTHON
        %pythoncode %{
            format = property(GetFormat, SetFormat)
            options = property(GetOptions, SetOptions)
        %}
#endif
}
