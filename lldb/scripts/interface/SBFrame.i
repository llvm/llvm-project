//===-- SWIG Interface for SBFrame ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents one of the stack frames associated with a thread.
SBThread contains SBFrame(s). For example (from test/lldbutil.py),

def print_stacktrace(thread, string_buffer = False):
    '''Prints a simple stack trace of this thread.'''

    ...

    for i in range(depth):
        frame = thread.GetFrameAtIndex(i)
        function = frame.GetFunction()

        load_addr = addrs[i].GetLoadAddress(target)
        if not function:
            file_addr = addrs[i].GetFileAddress()
            start_addr = frame.GetSymbol().GetStartAddress().GetFileAddress()
            symbol_offset = file_addr - start_addr
            print >> output, '  frame #{num}: {addr:#016x} {mod}`{symbol} + {offset}'.format(
                num=i, addr=load_addr, mod=mods[i], symbol=symbols[i], offset=symbol_offset)
        else:
            print >> output, '  frame #{num}: {addr:#016x} {mod}`{func} at {file}:{line} {args}'.format(
                num=i, addr=load_addr, mod=mods[i],
                func='%s [inlined]' % funcs[i] if frame.IsInlined() else funcs[i],
                file=files[i], line=lines[i],
                args=get_args_as_string(frame, showFuncName=False) if not frame.IsInlined() else '()')

    ...

And,

    for frame in thread:
        print frame

See also SBThread."
) SBFrame;
class SBFrame
{
public:
    SBFrame ();

    SBFrame (const lldb::SBFrame &rhs);
    
   ~SBFrame();

    bool
    IsEqual (const lldb::SBFrame &rhs) const;

    bool
    IsValid() const;

    uint32_t
    GetFrameID () const;

    %feature("docstring", "
    Get the Canonical Frame Address for this stack frame.
    This is the DWARF standard's definition of a CFA, a stack address
    that remains constant throughout the lifetime of the function.
    Returns an lldb::addr_t stack address, or LLDB_INVALID_ADDRESS if
    the CFA cannot be determined.") GetCFA;
    lldb::addr_t
    GetCFA () const;

    lldb::addr_t
    GetPC () const;

    bool
    SetPC (lldb::addr_t new_pc);

    lldb::addr_t
    GetSP () const;

    lldb::addr_t
    GetFP () const;

    lldb::SBAddress
    GetPCAddress () const;

    lldb::SBSymbolContext
    GetSymbolContext (uint32_t resolve_scope) const;

    lldb::SBModule
    GetModule () const;

    lldb::SBCompileUnit
    GetCompileUnit () const;

    lldb::SBFunction
    GetFunction () const;

    lldb::SBSymbol
    GetSymbol () const;

    %feature("docstring", "
    /// Gets the deepest block that contains the frame PC.
    ///
    /// See also GetFrameBlock().
    ") GetBlock;
    lldb::SBBlock
    GetBlock () const;

    %feature("docstring", "
    /// Get the appropriate function name for this frame. Inlined functions in
    /// LLDB are represented by Blocks that have inlined function information, so
    /// just looking at the SBFunction or SBSymbol for a frame isn't enough.
    /// This function will return the appropriate function, symbol or inlined
    /// function name for the frame.
    ///
    /// This function returns:
    /// - the name of the inlined function (if there is one)
    /// - the name of the concrete function (if there is one)
    /// - the name of the symbol (if there is one)
    /// - NULL
    ///
    /// See also IsInlined().
    ") GetFunctionName;
    const char *
    GetFunctionName();
             
     const char *
     GetDisplayFunctionName ();

    const char *
    GetFunctionName() const;
             
    %feature("docstring", "
    /// Returns the language of the frame's SBFunction, or if there.
    /// is no SBFunction, guess the language from the mangled name.
    /// .
    ") GuessLanguage;
    lldb::LanguageType
    GuessLanguage() const;

    bool IsSwiftThunk() const;

   %feature("docstring", "
    /// Return true if this frame represents an inlined function.
    ///
    /// See also GetFunctionName().
    ") IsInlined;
    bool
    IsInlined();

    bool
    IsInlined() const;

    %feature("docstring", "
    /// Return true if this frame is artificial (e.g a frame synthesized to
    /// capture a tail call). Local variables may not be available in an artificial
    /// frame.
    ") IsArtificial;
    bool
    IsArtificial();

    bool
    IsArtificial() const;

    %feature("docstring", "
    /// The version that doesn't supply a 'use_dynamic' value will use the
    /// target's default.
    ") EvaluateExpression;
    lldb::SBValue
    EvaluateExpression (const char *expr);    

    lldb::SBValue
    EvaluateExpression (const char *expr, lldb::DynamicValueType use_dynamic);

    lldb::SBValue
    EvaluateExpression (const char *expr, lldb::DynamicValueType use_dynamic, bool unwind_on_error);
    
    lldb::SBValue
    EvaluateExpression (const char *expr, SBExpressionOptions &options);

    %feature("docstring", "
    /// Gets the lexical block that defines the stack frame. Another way to think
    /// of this is it will return the block that contains all of the variables
    /// for a stack frame. Inlined functions are represented as SBBlock objects
    /// that have inlined function information: the name of the inlined function,
    /// where it was called from. The block that is returned will be the first 
    /// block at or above the block for the PC (SBFrame::GetBlock()) that defines
    /// the scope of the frame. When a function contains no inlined functions,
    /// this will be the top most lexical block that defines the function. 
    /// When a function has inlined functions and the PC is currently
    /// in one of those inlined functions, this method will return the inlined
    /// block that defines this frame. If the PC isn't currently in an inlined
    /// function, the lexical block that defines the function is returned.
    ") GetFrameBlock;
    lldb::SBBlock
    GetFrameBlock () const;

    lldb::SBLineEntry
    GetLineEntry () const;

    lldb::SBThread
    GetThread () const;

    const char *
    Disassemble () const;

    void
    Clear();

#ifndef SWIG
    bool
    operator == (const lldb::SBFrame &rhs) const;

    bool
    operator != (const lldb::SBFrame &rhs) const;

#endif

    %feature("docstring", "
    /// The version that doesn't supply a 'use_dynamic' value will use the
    /// target's default.
    ") GetVariables;
    lldb::SBValueList
    GetVariables (bool arguments,
                  bool locals,
                  bool statics,
                  bool in_scope_only);

    lldb::SBValueList
    GetVariables (bool arguments,
                  bool locals,
                  bool statics,
                  bool in_scope_only,
                  lldb::DynamicValueType  use_dynamic);

    lldb::SBValueList
    GetVariables (const lldb::SBVariablesOptions& options);
             
    lldb::SBValueList
    GetRegisters ();

    %feature("docstring", "
    /// The version that doesn't supply a 'use_dynamic' value will use the
    /// target's default.
    ") FindVariable;
    lldb::SBValue
    FindVariable (const char *var_name);

    lldb::SBValue
    FindVariable (const char *var_name, lldb::DynamicValueType use_dynamic);

    lldb::SBValue
    FindRegister (const char *name);

    %feature("docstring", "
    /// Get a lldb.SBValue for a variable path. 
    ///
    /// Variable paths can include access to pointer or instance members:
    ///     rect_ptr->origin.y
    ///     pt.x
    /// Pointer dereferences:
    ///     *this->foo_ptr
    ///     **argv
    /// Address of:
    ///     &pt
    ///     &my_array[3].x
    /// Array accesses and treating pointers as arrays:
    ///     int_array[1]
    ///     pt_ptr[22].x
    ///
    /// Unlike EvaluateExpression() which returns lldb.SBValue objects
    /// with constant copies of the values at the time of evaluation,
    /// the result of this function is a value that will continue to
    /// track the current value of the value as execution progresses
    /// in the current frame.
    ") GetValueForVariablePath;
    lldb::SBValue
    GetValueForVariablePath (const char *var_path);
             
    lldb::SBValue
    GetValueForVariablePath (const char *var_path, lldb::DynamicValueType use_dynamic);

    %feature("docstring", "
    /// Find variables, register sets, registers, or persistent variables using
    /// the frame as the scope.
    ///
    /// The version that doesn't supply a 'use_dynamic' value will use the
    /// target's default.
    ") FindValue;
    lldb::SBValue
    FindValue (const char *name, ValueType value_type);

    lldb::SBValue
    FindValue (const char *name, ValueType value_type, lldb::DynamicValueType use_dynamic);

    bool
    GetDescription (lldb::SBStream &description);
    
    %pythoncode %{
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

        __swig_getmethods__["pc"] = GetPC
        __swig_setmethods__["pc"] = SetPC
        if _newclass: pc = property(GetPC, SetPC)

        __swig_getmethods__["addr"] = GetPCAddress
        if _newclass: addr = property(GetPCAddress, None, doc='''A read only property that returns the program counter (PC) as a section offset address (lldb.SBAddress).''')

        __swig_getmethods__["fp"] = GetFP
        if _newclass: fp = property(GetFP, None, doc='''A read only property that returns the frame pointer (FP) as an unsigned integer.''')

        __swig_getmethods__["sp"] = GetSP
        if _newclass: sp = property(GetSP, None, doc='''A read only property that returns the stack pointer (SP) as an unsigned integer.''')

        __swig_getmethods__["module"] = GetModule
        if _newclass: module = property(GetModule, None, doc='''A read only property that returns an lldb object that represents the module (lldb.SBModule) for this stack frame.''')

        __swig_getmethods__["compile_unit"] = GetCompileUnit
        if _newclass: compile_unit = property(GetCompileUnit, None, doc='''A read only property that returns an lldb object that represents the compile unit (lldb.SBCompileUnit) for this stack frame.''')

        __swig_getmethods__["function"] = GetFunction
        if _newclass: function = property(GetFunction, None, doc='''A read only property that returns an lldb object that represents the function (lldb.SBFunction) for this stack frame.''')

        __swig_getmethods__["symbol"] = GetSymbol
        if _newclass: symbol = property(GetSymbol, None, doc='''A read only property that returns an lldb object that represents the symbol (lldb.SBSymbol) for this stack frame.''')

        __swig_getmethods__["block"] = GetBlock
        if _newclass: block = property(GetBlock, None, doc='''A read only property that returns an lldb object that represents the block (lldb.SBBlock) for this stack frame.''')

        __swig_getmethods__["is_inlined"] = IsInlined
        if _newclass: is_inlined = property(IsInlined, None, doc='''A read only property that returns an boolean that indicates if the block frame is an inlined function.''')

        __swig_getmethods__["name"] = GetFunctionName
        if _newclass: name = property(GetFunctionName, None, doc='''A read only property that retuns the name for the function that this frame represents. Inlined stack frame might have a concrete function that differs from the name of the inlined function (a named lldb.SBBlock).''')

        __swig_getmethods__["line_entry"] = GetLineEntry
        if _newclass: line_entry = property(GetLineEntry, None, doc='''A read only property that returns an lldb object that represents the line table entry (lldb.SBLineEntry) for this stack frame.''')

        __swig_getmethods__["thread"] = GetThread
        if _newclass: thread = property(GetThread, None, doc='''A read only property that returns an lldb object that represents the thread (lldb.SBThread) for this stack frame.''')

        __swig_getmethods__["disassembly"] = Disassemble
        if _newclass: disassembly = property(Disassemble, None, doc='''A read only property that returns the disassembly for this stack frame as a python string.''')

        __swig_getmethods__["idx"] = GetFrameID
        if _newclass: idx = property(GetFrameID, None, doc='''A read only property that returns the zero based stack frame index.''')

        __swig_getmethods__["variables"] = get_all_variables
        if _newclass: variables = property(get_all_variables, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the variables in this stack frame.''')

        __swig_getmethods__["vars"] = get_all_variables
        if _newclass: vars = property(get_all_variables, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the variables in this stack frame.''')

        __swig_getmethods__["locals"] = get_locals
        if _newclass: locals = property(get_locals, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the local variables in this stack frame.''')

        __swig_getmethods__["args"] = get_arguments
        if _newclass: args = property(get_arguments, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the argument variables in this stack frame.''')

        __swig_getmethods__["arguments"] = get_arguments
        if _newclass: arguments = property(get_arguments, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the argument variables in this stack frame.''')

        __swig_getmethods__["statics"] = get_statics
        if _newclass: statics = property(get_statics, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the static variables in this stack frame.''')

        __swig_getmethods__["registers"] = GetRegisters
        if _newclass: registers = property(GetRegisters, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the CPU registers for this stack frame.''')

        __swig_getmethods__["regs"] = GetRegisters
        if _newclass: regs = property(GetRegisters, None, doc='''A read only property that returns a list() that contains a collection of lldb.SBValue objects that represent the CPU registers for this stack frame.''')

        __swig_getmethods__["register"] = get_registers_access
        if _newclass: register = property(get_registers_access, None, doc='''A read only property that returns an helper object providing a flattened indexable view of the CPU registers for this stack frame.''')

        __swig_getmethods__["reg"] = get_registers_access
        if _newclass: reg = property(get_registers_access, None, doc='''A read only property that returns an helper object providing a flattened indexable view of the CPU registers for this stack frame''')

        __swig_getmethods__["parent"] = get_parent_frame
        if _newclass: parent = property(get_parent_frame, None, doc='''A read only property that returns the parent (caller) frame of the current frame.''')

    %}
};

} // namespace lldb
