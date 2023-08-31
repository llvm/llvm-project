STRING_EXTENSION_OUTSIDE(SBFunction)

%extend lldb::SBFunction {
#ifdef SWIGPYTHON
    %pythoncode %{
        def get_instructions_from_current_target (self):
            return self.GetInstructions (target)

        addr = property(GetStartAddress, None, doc='''A read only property that returns an lldb object that represents the start address (lldb.SBAddress) for this function.''')
        end_addr = property(GetEndAddress, None, doc='''A read only property that returns an lldb object that represents the end address (lldb.SBAddress) for this function.''')
        block = property(GetBlock, None, doc='''A read only property that returns an lldb object that represents the top level lexical block (lldb.SBBlock) for this function.''')
        instructions = property(get_instructions_from_current_target, None, doc='''A read only property that returns an lldb object that represents the instructions (lldb.SBInstructionList) for this function.''')
        mangled = property(GetMangledName, None, doc='''A read only property that returns the mangled (linkage) name for this function as a string.''')
        name = property(GetName, None, doc='''A read only property that returns the name for this function as a string.''')
        prologue_size = property(GetPrologueByteSize, None, doc='''A read only property that returns the size in bytes of the prologue instructions as an unsigned integer.''')
        type = property(GetType, None, doc='''A read only property that returns an lldb object that represents the return type (lldb.SBType) for this function.''')
    %}
#endif
}
