//===-- SWIG Interface for SBInstructionList --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

namespace lldb {

%feature("docstring",
"Represents a list of machine instructions.  SBFunction and SBSymbol have
GetInstructions() methods which return SBInstructionList instances.

SBInstructionList supports instruction (SBInstruction instance) iteration.
For example (see also SBDebugger for a more complete example),

def disassemble_instructions (insts):
    for i in insts:
        print i

defines a function which takes an SBInstructionList instance and prints out
the machine instructions in assembly format."
) SBInstructionList;
class SBInstructionList
{
public:

    SBInstructionList ();

    SBInstructionList (const SBInstructionList &rhs);
    
    ~SBInstructionList ();

    bool
    IsValid () const;

    size_t
    GetSize ();

    lldb::SBInstruction
    GetInstructionAtIndex (uint32_t idx);

    size_t GetInstructionsCount(const SBAddress &start, const SBAddress &end,
                                bool canSetBreakpoint);

    void
    Clear ();

    void
    AppendInstruction (lldb::SBInstruction inst);

    void
    Print (FILE *out);

    bool
    GetDescription (lldb::SBStream &description);
    
    bool
    DumpEmulationForAllInstructions (const char *triple);

    %pythoncode %{
        def __len__(self):
            '''Access len of the instruction list.'''
            return int(self.GetSize())

        def __getitem__(self, key):
            '''Access instructions by integer index for array access or by lldb.SBAddress to find an instruction that matches a section offset address object.'''
            if type(key) is int:
                # Find an instruction by index
                if key < len(self):
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

};

} // namespace lldb
