//===-- SWIG Interface for SBInstruction ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

// There's a lot to be fixed here, but need to wait for underlying insn implementation
// to be revised & settle down first.

namespace lldb {

class SBInstruction
{
public:

    SBInstruction ();

    SBInstruction (const SBInstruction &rhs);
    
    ~SBInstruction ();

    bool
    IsValid();

    lldb::SBAddress
    GetAddress();

    lldb::AddressClass
    GetAddressClass ();
    
    const char *
    GetMnemonic (lldb::SBTarget target);
    
    const char *
    GetOperands (lldb::SBTarget target);
    
    const char *
    GetComment (lldb::SBTarget target);
    
    lldb::SBData
    GetData (lldb::SBTarget target);
    
    size_t
    GetByteSize ();

    bool
    DoesBranch ();

    bool
    HasDelaySlot ();

    void
    Print (FILE *out);

    bool
    GetDescription (lldb::SBStream &description);

    bool
    EmulateWithFrame (lldb::SBFrame &frame, uint32_t evaluate_options);

    bool
    DumpEmulation (const char * triple); // triple is to specify the architecture, e.g. 'armv6' or 'armv7-apple-ios'
    
    bool
    TestEmulation (lldb::SBStream &output_stream, const char *test_file);
    
    %pythoncode %{
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

        __swig_getmethods__["mnemonic"] = __mnemonic_property__
        if _newclass: mnemonic = property(__mnemonic_property__, None, doc='''A read only property that returns the mnemonic for this instruction as a string.''')

        __swig_getmethods__["operands"] = __operands_property__
        if _newclass: operands = property(__operands_property__, None, doc='''A read only property that returns the operands for this instruction as a string.''')

        __swig_getmethods__["comment"] = __comment_property__
        if _newclass: comment = property(__comment_property__, None, doc='''A read only property that returns the comment for this instruction as a string.''')

        __swig_getmethods__["addr"] = GetAddress
        if _newclass: addr = property(GetAddress, None, doc='''A read only property that returns an lldb object that represents the address (lldb.SBAddress) for this instruction.''')
        
        __swig_getmethods__["size"] = GetByteSize
        if _newclass: size = property(GetByteSize, None, doc='''A read only property that returns the size in bytes for this instruction as an integer.''')

        __swig_getmethods__["is_branch"] = DoesBranch
        if _newclass: is_branch = property(DoesBranch, None, doc='''A read only property that returns a boolean value that indicates if this instruction is a branch instruction.''')
    %}
    

};

} // namespace lldb
