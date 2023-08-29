STRING_EXTENSION_OUTSIDE(SBFrame)

%extend lldb::SBFrame {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __eq__(self, other):
            return not self.__ne__(other)

        def __int__(self):
            pass

        def __hex__(self):
            pass

        def __oct__(self):
            pass

        def __len__(self):
            pass

        def __iter__(self):
            pass

        def get_all_variables(self):
            return self.GetVariables(True,True,True,True)

        def get_parent_frame(self):
            parent_idx = self.idx + 1
            if parent_idx >= 0 and parent_idx < len(self.thread.frame):
                return self.thread.frame[parent_idx]
            else:
                return SBFrame()

        def get_arguments(self):
            return self.GetVariables(True,False,False,False)

        def get_locals(self):
            return self.GetVariables(False,True,False,False)

        def get_statics(self):
            return self.GetVariables(False,False,True,False)

        def var(self, var_expr_path):
            '''Calls through to lldb.SBFrame.GetValueForVariablePath() and returns
            a value that represents the variable expression path'''
            return self.GetValueForVariablePath(var_expr_path)

        def get_registers_access(self):
            class registers_access(object):
                '''A helper object that exposes a flattened view of registers, masking away the notion of register sets for easy scripting.'''
                def __init__(self, regs):
                    self.regs = regs

                def __getitem__(self, key):
                    if type(key) is str:
                        for i in range(0,len(self.regs)):
                            rs = self.regs[i]
                            for j in range (0,rs.num_children):
                                reg = rs.GetChildAtIndex(j)
                                if reg.name == key: return reg
                    else:
                        return lldb.SBValue()

            return registers_access(self.registers)

        pc = property(GetPC, SetPC)
        addr = property(GetPCAddress, None, doc='''A read only property that returns the program counter (PC) as a section offset address (lldb.SBAddress).''')
        fp = property(GetFP, None, doc='''A read only property that returns the frame pointer (FP) as an unsigned integer.''')
        sp = property(GetSP, None, doc='''A read only property that returns the stack pointer (SP) as an unsigned integer.''')
        module = property(GetModule, None, doc='''A read only property that returns an lldb object that represents the module (lldb.SBModule) for this stack frame.''')
        compile_unit = property(GetCompileUnit, None, doc='''A read only property that returns an lldb object that represents the compile unit (lldb.SBCompileUnit) for this stack frame.''')
        function = property(GetFunction, None, doc='''A read only property that returns an lldb object that represents the function (lldb.SBFunction) for this stack frame.''')
        symbol = property(GetSymbol, None, doc='''A read only property that returns an lldb object that represents the symbol (lldb.SBSymbol) for this stack frame.''')
        block = property(GetBlock, None, doc='''A read only property that returns an lldb object that represents the block (lldb.SBBlock) for this stack frame.''')
        is_inlined = property(IsInlined, None, doc='''A read only property that returns an boolean that indicates if the block frame is an inlined function.''')
        name = property(GetFunctionName, None, doc='''A read only property that retuns the name for the function that this frame represents. Inlined stack frame might have a concrete function that differs from the name of the inlined function (a named lldb.SBBlock).''')
        line_entry = property(GetLineEntry, None, doc='''A read only property that returns an lldb object that represents the line table entry (lldb.SBLineEntry) for this stack frame.''')
        thread = property(GetThread, None, doc='''A read only property that returns an lldb object that represents the thread (lldb.SBThread) for this stack frame.''')
        disassembly = property(Disassemble, None, doc='''A read only property that returns the disassembly for this stack frame as a python string.''')
        idx = property(GetFrameID, None, doc='''A read only property that returns the zero based stack frame index.''')
        variables = property(get_all_variables, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the variables in this stack frame.''')
        vars = property(get_all_variables, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the variables in this stack frame.''')
        locals = property(get_locals, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the local variables in this stack frame.''')
        args = property(get_arguments, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the argument variables in this stack frame.''')
        arguments = property(get_arguments, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the argument variables in this stack frame.''')
        statics = property(get_statics, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the static variables in this stack frame.''')
        registers = property(GetRegisters, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the CPU registers for this stack frame.''')
        regs = property(GetRegisters, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the CPU registers for this stack frame.''')
        register = property(get_registers_access, None, doc='''A read only property that returns an helper object providing a flattened indexable view of the CPU registers for this stack frame.''')
        reg = property(get_registers_access, None, doc='''A read only property that returns an helper object providing a flattened indexable view of the CPU registers for this stack frame''')
        parent = property(get_parent_frame, None, doc='''A read only property that returns the parent (caller) frame of the current frame.''')
    %}
#endif
}
