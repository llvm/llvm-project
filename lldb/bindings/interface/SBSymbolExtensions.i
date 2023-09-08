STRING_EXTENSION_OUTSIDE(SBSymbol)

%extend lldb::SBSymbol {
#ifdef SWIGPYTHON
    %pythoncode %{
        # operator== is a free function, which swig does not handle, so we inject
        # our own equality operator here
        def __eq__(self, other):
            return not self.__ne__(other)

        def __hex__(self):
            return self.GetStartAddress()

        def get_instructions_from_current_target (self):
            return self.GetInstructions (target)

        name = property(GetName, None, doc='''A read only property that returns the name for this symbol as a string.''')
        mangled = property(GetMangledName, None, doc='''A read only property that returns the mangled (linkage) name for this symbol as a string.''')
        type = property(GetType, None, doc='''A read only property that returns an lldb enumeration value (see enumerations that start with "lldb.eSymbolType") that represents the type of this symbol.''')
        addr = property(GetStartAddress, None, doc='''A read only property that returns an lldb object that represents the start address (lldb.SBAddress) for this symbol.''')
        end_addr = property(GetEndAddress, None, doc='''A read only property that returns an lldb object that represents the end address (lldb.SBAddress) for this symbol.''')
        prologue_size = property(GetPrologueByteSize, None, doc='''A read only property that returns the size in bytes of the prologue instructions as an unsigned integer.''')
        instructions = property(get_instructions_from_current_target, None, doc='''A read only property that returns an lldb object that represents the instructions (lldb.SBInstructionList) for this symbol.''')
        external = property(IsExternal, None, doc='''A read only property that returns a boolean value that indicates if this symbol is externally visiable (exported) from the module that contains it.''')
        synthetic = property(IsSynthetic, None, doc='''A read only property that returns a boolean value that indicates if this symbol was synthetically created from information in module that contains it.''')
    %}
#endif
}
