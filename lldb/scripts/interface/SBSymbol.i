//===-- SWIG Interface for SBSymbol -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents the symbol possibly associated with a stack frame.
SBModule contains SBSymbol(s). SBSymbol can also be retrieved from SBFrame.

See also SBModule and SBFrame."
) SBSymbol;
class SBSymbol
{
public:

    SBSymbol ();

    ~SBSymbol ();

    SBSymbol (const lldb::SBSymbol &rhs);

    bool
    IsValid () const;

    explicit operator bool() const;


    const char *
    GetName() const;

    const char *
    GetDisplayName() const;

    const char *
    GetMangledName () const;

    lldb::SBInstructionList
    GetInstructions (lldb::SBTarget target);

    lldb::SBInstructionList
    GetInstructions (lldb::SBTarget target, const char *flavor_string);

    SBAddress
    GetStartAddress ();

    SBAddress
    GetEndAddress ();

    uint32_t
    GetPrologueByteSize ();

    SymbolType
    GetType ();

    bool
    GetDescription (lldb::SBStream &description);

    bool
    IsExternal();

    bool
    IsSynthetic();

    bool
    operator == (const lldb::SBSymbol &rhs) const;

    bool
    operator != (const lldb::SBSymbol &rhs) const;

#ifdef SWIGPYTHON
    %pythoncode %{
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

};

} // namespace lldb
