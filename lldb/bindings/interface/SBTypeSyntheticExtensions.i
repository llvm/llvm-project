STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeSynthetic, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeSynthetic {
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
            contains_code = property(IsClassCode, None)
            synthetic_data = property(GetData, None)
        %}
#endif
}
