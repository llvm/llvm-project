STRING_EXTENSION_OUTSIDE(SBInstructionList)

%extend lldb::SBInstructionList {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __iter__(self):
            '''Iterate over all instructions in a lldb.SBInstructionList
            object.'''
            return lldb_iter(self, 'GetSize', 'GetInstructionAtIndex')

        def __len__(self):
            '''Access len of the instruction list.'''
            return int(self.GetSize())

        def __getitem__(self, key):
            '''Access instructions by integer index for array access or by lldb.SBAddress to find an instruction that matches a section offset address object.'''
            if type(key) is int:
                # Find an instruction by index
                count = len(self)
                if -count <= key < count:
                    key %= count
                    return self.GetInstructionAtIndex(key)
            elif type(key) is SBAddress:
                # Find an instruction using a lldb.SBAddress object
                lookup_file_addr = key.file_addr
                closest_inst = None
                for idx in range(self.GetSize()):
                    inst = self.GetInstructionAtIndex(idx)
                    inst_file_addr = inst.addr.file_addr
                    if inst_file_addr == lookup_file_addr:
                        return inst
                    elif inst_file_addr > lookup_file_addr:
                        return closest_inst
                    else:
                        closest_inst = inst
            return None
    %}
#endif
}
