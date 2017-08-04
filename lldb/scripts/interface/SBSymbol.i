//===-- SWIG Interface for SBSymbol -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents the symbol possibly associated with a stack frame.
SBModule contains SBSymbol(s). SBSymbol can also be retrived from SBFrame.

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
    
    %pythoncode %{
        def get_instructions_from_current_target (self):
            return self.GetInstructions (target)
        
        __swig_getmethods__["name"] = GetName
        if _newclass: name = property(GetName, None, doc='''A read only property that returns the name for this symbol as a string.''')
        
        __swig_getmethods__["mangled"] = GetMangledName
        if _newclass: mangled = property(GetMangledName, None, doc='''A read only property that returns the mangled (linkage) name for this symbol as a string.''')
        
        __swig_getmethods__["type"] = GetType
        if _newclass: type = property(GetType, None, doc='''A read only property that returns an lldb enumeration value (see enumerations that start with "lldb.eSymbolType") that represents the type of this symbol.''')
        
        __swig_getmethods__["addr"] = GetStartAddress
        if _newclass: addr = property(GetStartAddress, None, doc='''A read only property that returns an lldb object that represents the start address (lldb.SBAddress) for this symbol.''')
        
        __swig_getmethods__["end_addr"] = GetEndAddress
        if _newclass: end_addr = property(GetEndAddress, None, doc='''A read only property that returns an lldb object that represents the end address (lldb.SBAddress) for this symbol.''')
        
        __swig_getmethods__["prologue_size"] = GetPrologueByteSize
        if _newclass: prologue_size = property(GetPrologueByteSize, None, doc='''A read only property that returns the size in bytes of the prologue instructions as an unsigned integer.''')
        
        __swig_getmethods__["instructions"] = get_instructions_from_current_target
        if _newclass: instructions = property(get_instructions_from_current_target, None, doc='''A read only property that returns an lldb object that represents the instructions (lldb.SBInstructionList) for this symbol.''')

        __swig_getmethods__["external"] = IsExternal
        if _newclass: external = property(IsExternal, None, doc='''A read only property that returns a boolean value that indicates if this symbol is externally visiable (exported) from the module that contains it.''')

        __swig_getmethods__["synthetic"] = IsSynthetic
        if _newclass: synthetic = property(IsSynthetic, None, doc='''A read only property that returns a boolean value that indicates if this symbol was synthetically created from information in module that contains it.''')

        
    %}

};

} // namespace lldb
