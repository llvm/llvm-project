//===-- SWIG Interface for SBSymbolContext ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"A context object that provides access to core debugger entities.

Many debugger functions require a context when doing lookups. This class
provides a common structure that can be used as the result of a query that
can contain a single result.

For example,

        exe = os.path.join(os.getcwd(), 'a.out')

        # Create a target for the debugger.
        target = self.dbg.CreateTarget(exe)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        # The inferior should stop on 'c'.
        from lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)

        # Now get the SBSymbolContext from this frame.  We want everything. :-)
        context = frame0.GetSymbolContext(lldb.eSymbolContextEverything)

        # Get the module.
        module = context.GetModule()
        ...

        # And the compile unit associated with the frame.
        compileUnit = context.GetCompileUnit()
        ...
"
) SBSymbolContext;
class SBSymbolContext
{
public:
    SBSymbolContext ();

    SBSymbolContext (const lldb::SBSymbolContext& rhs);

    ~SBSymbolContext ();

    bool
    IsValid () const;

    explicit operator bool() const;

    lldb::SBModule        GetModule ();
    lldb::SBCompileUnit   GetCompileUnit ();
    lldb::SBFunction      GetFunction ();
    lldb::SBBlock         GetBlock ();
    lldb::SBLineEntry     GetLineEntry ();
    lldb::SBSymbol        GetSymbol ();

    void SetModule      (lldb::SBModule module);
    void SetCompileUnit (lldb::SBCompileUnit compile_unit);
    void SetFunction    (lldb::SBFunction function);
    void SetBlock       (lldb::SBBlock block);
    void SetLineEntry   (lldb::SBLineEntry line_entry);
    void SetSymbol      (lldb::SBSymbol symbol);

    lldb::SBSymbolContext
    GetParentOfInlinedScope (const lldb::SBAddress &curr_frame_pc,
                             lldb::SBAddress &parent_frame_addr) const;


    bool
    GetDescription (lldb::SBStream &description);

    STRING_EXTENSION(SBSymbolContext)

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

};

} // namespace lldb
