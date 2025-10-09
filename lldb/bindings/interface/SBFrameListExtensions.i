%extend lldb::SBFrameList {

#ifdef SWIGPYTHON
       %nothreadallow;
#endif
       std::string lldb::SBFrameList::__str__ (){
           lldb::SBStream description;
           const size_t n = $self->GetSize();
           if (n)
           {
               for (size_t i=0; i<n; ++i)
                   $self->GetFrameAtIndex(i).GetDescription(description);
           }
           else
           {
               description.Printf("<empty> lldb.SBFrameList()");
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
            '''Iterate over all frames in a lldb.SBFrameList object.'''
            return lldb_iter(self, 'GetSize', 'GetFrameAtIndex')

        def __len__(self):
            return int(self.GetSize())

        def __getitem__(self, key):
            count = len(self)
            if type(key) is int:
                if -count <= key < count:
                    key %= count
                    return self.GetFrameAtIndex(key)
            return None
    %}
#endif
}