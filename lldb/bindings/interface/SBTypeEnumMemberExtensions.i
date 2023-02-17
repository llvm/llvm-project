STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeEnumMember, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeEnumMember {
#ifdef SWIGPYTHON
    %pythoncode %{
        name = property(GetName, None, doc='''A read only property that returns the name for this enum member as a string.''')
        type = property(GetType, None, doc='''A read only property that returns an lldb object that represents the type (lldb.SBType) for this enum member.''')
        signed = property(GetValueAsSigned, None, doc='''A read only property that returns the value of this enum member as a signed integer.''')
        unsigned = property(GetValueAsUnsigned, None, doc='''A read only property that returns the value of this enum member as a unsigned integer.''')
    %}
#endif
}

%extend lldb::SBTypeEnumMemberList {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __iter__(self):
            '''Iterate over all members in a lldb.SBTypeEnumMemberList object.'''
            return lldb_iter(self, 'GetSize', 'GetTypeEnumMemberAtIndex')

        def __len__(self):
            '''Return the number of members in a lldb.SBTypeEnumMemberList object.'''
            return self.GetSize()

        def __getitem__(self, key):
          num_elements = self.GetSize()
          if type(key) is int:
              if -num_elements <= key < num_elements:
                  key %= num_elements
                  return self.GetTypeEnumMemberAtIndex(key)
          elif type(key) is str:
              for idx in range(num_elements):
                  item = self.GetTypeEnumMemberAtIndex(idx)
                  if item.name == key:
                      return item
          return None
    %}
#endif
}
