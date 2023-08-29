STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeNameSpecifier, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeNameSpecifier {
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

            name = property(GetName)
            is_regex = property(IsRegex)
        %}
#endif
}
