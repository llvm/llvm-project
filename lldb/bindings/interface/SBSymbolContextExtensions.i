STRING_EXTENSION_OUTSIDE(SBSymbolContext)

%extend lldb::SBSymbolContext {
#ifdef SWIGPYTHON
    %pythoncode %{
        module = property(GetModule, SetModule, doc='''A read/write property that allows the getting/setting of the module (lldb.SBModule) in this symbol context.''')
        compile_unit = property(GetCompileUnit, SetCompileUnit, doc='''A read/write property that allows the getting/setting of the compile unit (lldb.SBCompileUnit) in this symbol context.''')
        function = property(GetFunction, SetFunction, doc='''A read/write property that allows the getting/setting of the function (lldb.SBFunction) in this symbol context.''')
        block = property(GetBlock, SetBlock, doc='''A read/write property that allows the getting/setting of the block (lldb.SBBlock) in this symbol context.''')
        symbol = property(GetSymbol, SetSymbol, doc='''A read/write property that allows the getting/setting of the symbol (lldb.SBSymbol) in this symbol context.''')
        line_entry = property(GetLineEntry, SetLineEntry, doc='''A read/write property that allows the getting/setting of the line entry (lldb.SBLineEntry) in this symbol context.''')
    %}
#endif
}
