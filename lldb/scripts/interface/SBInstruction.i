//===-- SWIG Interface for SBInstruction ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

    explicit operator bool() const;

    lldb::SBAddress
    GetAddress();


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

    bool
    CanSetBreakpoint ();

    void
    Print (lldb::SBFile out);

    void
    Print (lldb::FileSP BORROWED);

    bool
    GetDescription (lldb::SBStream &description);

    bool
    EmulateWithFrame (lldb::SBFrame &frame, uint32_t evaluate_options);

    bool
    DumpEmulation (const char * triple); // triple is to specify the architecture, e.g. 'armv6' or 'armv7-apple-ios'

    bool
    TestEmulation (lldb::SBStream &output_stream, const char *test_file);

    STRING_EXTENSION(SBInstruction)

#ifdef SWIGPYTHON
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

        mnemonic = property(__mnemonic_property__, None, doc='''A read only property that returns the mnemonic for this instruction as a string.''')
        operands = property(__operands_property__, None, doc='''A read only property that returns the operands for this instruction as a string.''')
        comment = property(__comment_property__, None, doc='''A read only property that returns the comment for this instruction as a string.''')
        addr = property(GetAddress, None, doc='''A read only property that returns an lldb object that represents the address (lldb.SBAddress) for this instruction.''')
        size = property(GetByteSize, None, doc='''A read only property that returns the size in bytes for this instruction as an integer.''')
        is_branch = property(DoesBranch, None, doc='''A read only property that returns a boolean value that indicates if this instruction is a branch instruction.''')
    %}
#endif


};

} // namespace lldb
