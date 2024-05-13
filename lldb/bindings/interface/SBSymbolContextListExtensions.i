STRING_EXTENSION_OUTSIDE(SBSymbolContextList)

%extend lldb::SBSymbolContextList {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __iter__(self):
            '''Iterate over all symbol contexts in a lldb.SBSymbolContextList
            object.'''
            return lldb_iter(self, 'GetSize', 'GetContextAtIndex')

        def __len__(self):
            return int(self.GetSize())

        def __getitem__(self, key):
            count = len(self)
            if isinstance(key, int):
                if -count <= key < count:
                    key %= count
                    return self.GetContextAtIndex(key)
                else:
                    raise IndexError
            raise TypeError

        def get_module_array(self):
            a = []
            for i in range(len(self)):
                obj = self.GetContextAtIndex(i).module
                if obj:
                    a.append(obj)
            return a

        def get_compile_unit_array(self):
            a = []
            for i in range(len(self)):
                obj = self.GetContextAtIndex(i).compile_unit
                if obj:
                    a.append(obj)
            return a
        def get_function_array(self):
            a = []
            for i in range(len(self)):
                obj = self.GetContextAtIndex(i).function
                if obj:
                    a.append(obj)
            return a
        def get_block_array(self):
            a = []
            for i in range(len(self)):
                obj = self.GetContextAtIndex(i).block
                if obj:
                    a.append(obj)
            return a
        def get_symbol_array(self):
            a = []
            for i in range(len(self)):
                obj = self.GetContextAtIndex(i).symbol
                if obj:
                    a.append(obj)
            return a
        def get_line_entry_array(self):
            a = []
            for i in range(len(self)):
                obj = self.GetContextAtIndex(i).line_entry
                if obj:
                    a.append(obj)
            return a

        modules = property(get_module_array, None, doc='''Returns a list() of lldb.SBModule objects, one for each module in each SBSymbolContext object in this list.''')
        compile_units = property(get_compile_unit_array, None, doc='''Returns a list() of lldb.SBCompileUnit objects, one for each compile unit in each SBSymbolContext object in this list.''')
        functions = property(get_function_array, None, doc='''Returns a list() of lldb.SBFunction objects, one for each function in each SBSymbolContext object in this list.''')
        blocks = property(get_block_array, None, doc='''Returns a list() of lldb.SBBlock objects, one for each block in each SBSymbolContext object in this list.''')
        line_entries = property(get_line_entry_array, None, doc='''Returns a list() of lldb.SBLineEntry objects, one for each line entry in each SBSymbolContext object in this list.''')
        symbols = property(get_symbol_array, None, doc='''Returns a list() of lldb.SBSymbol objects, one for each symbol in each SBSymbolContext object in this list.''')
    %}
#endif
}
