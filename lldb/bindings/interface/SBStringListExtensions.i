%extend lldb::SBStringList {
    std::string __repr__() {
        const uint32_t size = $self->GetSize(); 
        if (size == 0) {
            return {"[]"};
        }

        std::string result;
        std::string_view separator;

        result += '[';
        for (uint32_t i = 0; i < size ; ++i) {
            result += std::exchange(separator, ", ");
            auto item = std::string_view($self->GetStringAtIndex(i));
            result += '\'';
            result += item;
            result += '\'';
        }
        result += ']';

        return result;
    }

#ifdef SWIGPYTHON
    %pythoncode%{
    def __iter__(self):
        '''Iterate over all strings in a lldb.SBStringList object.'''
        for i in range(self.GetSize()):
            yield self.GetStringAtIndex(i)

    def __len__(self):
        '''Return the number of strings in a lldb.SBStringList object.'''
        return self.GetSize()

    def __getitem__(self, subscript: 'int | slice[int | None]', /):
        if isinstance(subscript, int):
            size = self.GetSize()
            idx = subscript

            if idx < 0:
                idx += size
            if idx >= size:
                raise IndexError("SBStringList index out of range")
            return self.GetStringAtIndex(idx)

        if isinstance(subscript, slice):
            indices = subscript.indices(self.GetSize())
            return [self.GetStringAtIndex(i) for i in range(*indices)]

        raise TypeError(f"SBStringList indices must be integers or slices, not {type(subscript).__name__}") 

    %}
#endif
}
