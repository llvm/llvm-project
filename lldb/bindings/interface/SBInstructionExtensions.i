STRING_EXTENSION_OUTSIDE(SBInstruction)

%extend lldb::SBInstruction {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __hex__(self):
            """ Returns the address of the instruction. """
            return self.GetAddress()

        def __len__(self):
            """ Returns the size of the instruction. """
            return self.GetByteSize()

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

        def variable_annotations(self):
            """Get variable annotations as a Python list of dictionaries.

            Returns:
                List of dictionaries, each containing variable annotation data
            """
            structured_data = self.GetVariableAnnotations()
            if not structured_data.IsValid():
                return []

            annotations = []
            for i in range(structured_data.GetSize()):
                item = structured_data.GetItemAtIndex(i)
                if item.GetType() != eStructuredDataTypeDictionary:
                    continue

                annotation = {}

                integer_fields = ['start_address', 'end_address', 'register_kind', 'decl_line']
                string_fields = ['variable_name', 'location_description', 'decl_file', 'type_name']

                for field in integer_fields:
                    value = item.GetValueForKey(field)
                    if value.IsValid():
                        annotation[field] = value.GetUnsignedIntegerValue()

                for field in string_fields:
                    value = item.GetValueForKey(field)
                    if value.IsValid():
                        annotation[field] = value.GetStringValue(1024)

                annotations.append(annotation)

            return annotations


        mnemonic = property(__mnemonic_property__, None, doc='''A read only property that returns the mnemonic for this instruction as a string.''')
        operands = property(__operands_property__, None, doc='''A read only property that returns the operands for this instruction as a string.''')
        comment = property(__comment_property__, None, doc='''A read only property that returns the comment for this instruction as a string.''')
        addr = property(GetAddress, None, doc='''A read only property that returns an lldb object that represents the address (lldb.SBAddress) for this instruction.''')
        size = property(GetByteSize, None, doc='''A read only property that returns the size in bytes for this instruction as an integer.''')
        is_branch = property(DoesBranch, None, doc='''A read only property that returns a boolean value that indicates if this instruction is a branch instruction.''')
    %}
#endif
}
