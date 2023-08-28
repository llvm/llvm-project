STRING_EXTENSION_OUTSIDE(SBAddress)

%extend lldb::SBAddress {
#ifdef SWIGPYTHON
    // operator== is a free function, which swig does not handle, so we inject
    // our own equality operator here
    %pythoncode%{
    def __eq__(self, other):
      return not self.__ne__(other)

    def __len__(self):
        pass

    def __iter__(self):
        pass
    %}

    %pythoncode %{
        __runtime_error_str = 'This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'

        def __get_load_addr_property__ (self):
            '''Get the load address for a lldb.SBAddress using the current target. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'''
            if not target:
                raise RuntimeError(self.__runtime_error_str)
            return self.GetLoadAddress (target)

        def __set_load_addr_property__ (self, load_addr):
            '''Set the load address for a lldb.SBAddress using the current target. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'''
            if not target:
                raise RuntimeError(self.__runtime_error_str)
            return self.SetLoadAddress (load_addr, target)

        def __int__(self):
            '''Convert an address to a load address if there is a process and that process is alive, or to a file address otherwise. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'''
            if not process or not target:
                raise RuntimeError(self.__runtime_error_str)
            if process.is_alive:
                return self.GetLoadAddress (target)
            return self.GetFileAddress ()

        def __oct__(self):
            '''Convert the address to an octal string. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'''
            return '%o' % int(self)

        def __hex__(self):
            '''Convert the address to an hex string. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'''
            return '0x%x' % int(self)

        module = property(GetModule, None, doc='''A read only property that returns an lldb object that represents the module (lldb.SBModule) that this address resides within.''')
        compile_unit = property(GetCompileUnit, None, doc='''A read only property that returns an lldb object that represents the compile unit (lldb.SBCompileUnit) that this address resides within.''')
        line_entry = property(GetLineEntry, None, doc='''A read only property that returns an lldb object that represents the line entry (lldb.SBLineEntry) that this address resides within.''')
        function = property(GetFunction, None, doc='''A read only property that returns an lldb object that represents the function (lldb.SBFunction) that this address resides within.''')
        block = property(GetBlock, None, doc='''A read only property that returns an lldb object that represents the block (lldb.SBBlock) that this address resides within.''')
        symbol = property(GetSymbol, None, doc='''A read only property that returns an lldb object that represents the symbol (lldb.SBSymbol) that this address resides within.''')
        offset = property(GetOffset, None, doc='''A read only property that returns the section offset in bytes as an integer.''')
        section = property(GetSection, None, doc='''A read only property that returns an lldb object that represents the section (lldb.SBSection) that this address resides within.''')
        file_addr = property(GetFileAddress, None, doc='''A read only property that returns file address for the section as an integer. This is the address that represents the address as it is found in the object file that defines it.''')
        load_addr = property(__get_load_addr_property__, __set_load_addr_property__, doc='''A read/write property that gets/sets the SBAddress using load address. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.''')
    %}
#endif
}
