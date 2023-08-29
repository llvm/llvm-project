STRING_EXTENSION_OUTSIDE(SBInstruction)

%extend lldb::SBInstruction {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __eq__(self, other):
            return not self.__ne__(other)

        def __int__(self):
            pass

        def __hex__(self):
            """ Returns the address of the instruction. """
            return self.GetAddress()

        def __oct__(self):
            pass

        def __len__(self):
            """ Returns the size of the instruction. """
            return self.GetByteSize()

        def __iter__(self):
            pass

        def __mnemonic_property__ (self):
            return self.GetMnemonic (target)
        def __operands_property__ (self):
            return self.GetOperands (target)
        def __comment_property__ (self):
            return self.GetComment (target)
        def __file_addr_property__ (self):
            return self.GetAddress ().GetFileAddress()
        def __load_adrr_property__ (self):
            return self.GetComment (target)

        mnemonic = property(__mnemonic_property__, None, doc='''A read only property that returns the mnemonic for this instruction as a string.''')
        operands = property(__operands_property__, None, doc='''A read only property that returns the operands for this instruction as a string.''')
        comment = property(__comment_property__, None, doc='''A read only property that returns the comment for this instruction as a string.''')
        addr = property(GetAddress, None, doc='''A read only property that returns an lldb object that represents the address (lldb.SBAddress) for this instruction.''')
        size = property(GetByteSize, None, doc='''A read only property that returns the size in bytes for this instruction as an integer.''')
        is_branch = property(DoesBranch, None, doc='''A read only property that returns a boolean value that indicates if this instruction is a branch instruction.''')
    %}
#endif
}
