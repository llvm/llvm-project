%extend lldb::SBFrameList {

#ifdef SWIGPYTHON
       %nothreadallow;
#endif
       std::string lldb::SBFrameList::__str__ (){
           lldb::SBStream description;
           if (!$self->GetDescription(description))
               return std::string("<empty> lldb.SBFrameList()");
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
            if type(key) is not int:
                return None
            if key < 0:
                count = len(self)
                if -count <= key < count:
                    key %= count

            frame = self.GetFrameAtIndex(key)
            return frame if frame.IsValid() else None
    %}
#endif
}
