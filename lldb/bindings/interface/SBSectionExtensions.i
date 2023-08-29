STRING_EXTENSION_OUTSIDE(SBSection)

%extend lldb::SBSection {
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

        def __iter__(self):
            '''Iterate over all subsections in a lldb.SBSection object.'''
            return lldb_iter(self, 'GetNumSubSections', 'GetSubSectionAtIndex')

        def __len__(self):
            '''Return the number of subsections in a lldb.SBSection object.'''
            return self.GetNumSubSections()

        def get_addr(self):
            return SBAddress(self, 0)

        name = property(GetName, None, doc='''A read only property that returns the name of this section as a string.''')
        addr = property(get_addr, None, doc='''A read only property that returns an lldb object that represents the start address (lldb.SBAddress) for this section.''')
        file_addr = property(GetFileAddress, None, doc='''A read only property that returns an integer that represents the starting "file" address for this section, or the address of the section in the object file in which it is defined.''')
        size = property(GetByteSize, None, doc='''A read only property that returns the size in bytes of this section as an integer.''')
        file_offset = property(GetFileOffset, None, doc='''A read only property that returns the file offset in bytes of this section as an integer.''')
        file_size = property(GetFileByteSize, None, doc='''A read only property that returns the file size in bytes of this section as an integer.''')
        data = property(GetSectionData, None, doc='''A read only property that returns an lldb object that represents the bytes for this section (lldb.SBData) for this section.''')
        type = property(GetSectionType, None, doc='''A read only property that returns an lldb enumeration value (see enumerations that start with "lldb.eSectionType") that represents the type of this section (code, data, etc.).''')
        target_byte_size = property(GetTargetByteSize, None, doc='''A read only property that returns the size of a target byte represented by this section as a number of host bytes.''')
        alignment = property(GetAlignment, None, doc='''A read only property that returns the alignment of this section as a number of host bytes.''')
    %}
#endif
}
