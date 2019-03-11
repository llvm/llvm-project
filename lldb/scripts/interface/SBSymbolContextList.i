//===-- SWIG Interface for SBSymbolContextList ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a list of symbol context object. See also SBSymbolContext.

For example (from test/python_api/target/TestTargetAPI.py),

    def find_functions(self, exe_name):
        '''Exercise SBTaget.FindFunctions() API.'''
        exe = os.path.join(os.getcwd(), exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        list = lldb.SBSymbolContextList()
        num = target.FindFunctions('c', lldb.eFunctionNameTypeAuto, False, list)
        self.assertTrue(num == 1 and list.GetSize() == 1)

        for sc in list:
            self.assertTrue(sc.GetModule().GetFileSpec().GetFilename() == exe_name)
            self.assertTrue(sc.GetSymbol().GetName() == 'c')                
") SBSymbolContextList;
class SBSymbolContextList
{
public:
    SBSymbolContextList ();

    SBSymbolContextList (const lldb::SBSymbolContextList& rhs);

    ~SBSymbolContextList ();

    bool
    IsValid () const;

    explicit operator bool() const;

    uint32_t
    GetSize() const;

    SBSymbolContext
    GetContextAtIndex (uint32_t idx);

    void
    Append (lldb::SBSymbolContext &sc);
    
    void
    Append (lldb::SBSymbolContextList &sc_list);

    bool
    GetDescription (lldb::SBStream &description);

    void
    Clear();
    
    %pythoncode %{
        def __len__(self):
            return int(self.GetSize())

        def __getitem__(self, key):
            count = len(self)
            if type(key) is int:
                if key < count:
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
        __swig_getmethods__["modules"] = get_module_array
        if _newclass: modules = property(get_module_array, None, doc='''Returns a list() of lldb.SBModule objects, one for each module in each SBSymbolContext object in this list.''')
        
        __swig_getmethods__["compile_units"] = get_compile_unit_array
        if _newclass: compile_units = property(get_compile_unit_array, None, doc='''Returns a list() of lldb.SBCompileUnit objects, one for each compile unit in each SBSymbolContext object in this list.''')
        
        __swig_getmethods__["functions"] = get_function_array
        if _newclass: functions = property(get_function_array, None, doc='''Returns a list() of lldb.SBFunction objects, one for each function in each SBSymbolContext object in this list.''')
        
        __swig_getmethods__["blocks"] = get_block_array
        if _newclass: blocks = property(get_block_array, None, doc='''Returns a list() of lldb.SBBlock objects, one for each block in each SBSymbolContext object in this list.''')
        
        __swig_getmethods__["line_entries"] = get_line_entry_array
        if _newclass: line_entries = property(get_line_entry_array, None, doc='''Returns a list() of lldb.SBLineEntry objects, one for each line entry in each SBSymbolContext object in this list.''')
        
        __swig_getmethods__["symbols"] = get_symbol_array
        if _newclass: symbols = property(get_symbol_array, None, doc='''Returns a list() of lldb.SBSymbol objects, one for each symbol in each SBSymbolContext object in this list.''')
    %}

};

} // namespace lldb
