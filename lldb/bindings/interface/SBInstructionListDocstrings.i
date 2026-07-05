%feature("docstring",
"Represents a list of machine instructions.  SBFunction and SBSymbol have
GetInstructions() methods which return SBInstructionList instances.

SBInstructionList supports instruction (:py:class:`SBInstruction` instance) iteration.
For example (see also :py:class:`SBDebugger` for a more complete example), ::

    def disassemble_instructions (insts):
        for i in insts:
            print i

defines a function which takes an SBInstructionList instance and prints out
the machine instructions in assembly format."
) lldb::SBInstructionList;
