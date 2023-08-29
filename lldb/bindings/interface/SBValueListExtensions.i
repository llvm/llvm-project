%extend lldb::SBValueList {

#ifdef SWIGPYTHON
       %nothreadallow;
#endif
       std::string lldb::SBValueList::__str__ (){
           lldb::SBStream description;
           const size_t n = $self->GetSize();
           if (n)
           {
               for (size_t i=0; i<n; ++i)
                   $self->GetValueAtIndex(i).GetDescription(description);
           }
           else
           {
               description.Printf("<empty> lldb.SBValueList()");
           }
           const char *desc = description.GetData();
           size_t desc_len = description.GetSize();
           if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r'))
               --desc_len;
           return std::string(desc, desc_len);
       }
#ifdef SWIGPYTHON
       %clearnothreadallow;
#endif

#ifdef SWIGPYTHON
    %pythoncode %{
        def __iter__(self):
            '''Iterate over all values in a lldb.SBValueList object.'''
            return lldb_iter(self, 'GetSize', 'GetValueAtIndex')

        def __len__(self):
            return int(self.GetSize())

        def __eq__(self, other):
            return not self.__ne__(other)

        def __int__(self):
            pass

        def __hex__(self):
            pass

        def __oct__(self):
            pass

        def __getitem__(self, key):
            count = len(self)
            #------------------------------------------------------------
            # Access with "int" to get Nth item in the list
            #------------------------------------------------------------
            if type(key) is int:
                if -count <= key < count:
                    key %= count
                    return self.GetValueAtIndex(key)
            #------------------------------------------------------------
            # Access with "str" to get values by name
            #------------------------------------------------------------
            elif type(key) is str:
                matches = []
                for idx in range(count):
                    value = self.GetValueAtIndex(idx)
                    if value.name == key:
                        matches.append(value)
                return matches
            #------------------------------------------------------------
            # Match with regex
            #------------------------------------------------------------
            elif isinstance(key, type(re.compile('.'))):
                matches = []
                for idx in range(count):
                    value = self.GetValueAtIndex(idx)
                    re_match = key.search(value.name)
                    if re_match:
                        matches.append(value)
                return matches

    %}
#endif
}
