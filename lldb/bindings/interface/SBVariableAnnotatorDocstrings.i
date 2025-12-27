%feature("docstring",
"Provides variable annotation functionality for disassembly instructions.

The SBVariableAnnotator class enables retrieval of structured variable 
location information for assembly instructions during debugging. This 
allows tools to understand where variables are stored (registers, memory) 
at specific instruction addresses, which is essential for debugging 
optimized code where variables may move between locations.

The annotator extracts DWARF debug information to provide:
* Variable names and types
* Current storage location (register name, memory, undefined)
* Address ranges where the location is valid
* Source declaration information

For example, when debugging optimized code::

    annotator = lldb.SBVariableAnnotator()
    target = lldb.debugger.GetSelectedTarget()
    frame = target.GetProcess().GetSelectedThread().GetSelectedFrame()

    # Get instructions for current function
    function = frame.GetFunction()
    instructions = target.ReadInstructions(function.GetStartAddress(),
                                         function.GetEndAddress())

    # Annotate each instruction
    for i in range(instructions.GetSize()):
        inst = instructions.GetInstructionAtIndex(i)
        annotations = annotator.AnnotateStructured(inst)

        # Process structured annotation data
        for j in range(annotations.GetSize()):
            item = annotations.GetItemAtIndex(j)
            var_name = item.GetValueForKey('variable_name').GetStringValue(1024)
            location = item.GetValueForKey('location_description').GetStringValue(1024)
            is_live = item.GetValueForKey('is_live').GetBooleanValue()

            print(f'Variable {var_name} in {location}, live: {is_live}')"
) lldb::SBVariableAnnotator;

%feature("docstring", "
    Create a new variable annotator instance.

    The annotator can be used to extract variable location information
    from assembly instructions when debugging programs compiled with
    debug information.")
lldb::SBVariableAnnotator::SBVariableAnnotator;

%feature("docstring", "
    Get variable annotations for an instruction as structured data.

    Returns an SBStructuredData object containing an array of dictionaries,
    where each dictionary represents one variable annotation with the following keys:

    * 'variable_name' (string): Name of the variable
    * 'location_description' (string): Where the variable is stored
      (register name like 'RDI', 'R15', or 'undef' if not available)
    * 'is_live' (boolean): Whether the variable is live at this instruction
    * 'start_address' (integer): Address where this location becomes valid
    * 'end_address' (integer): Address where this location becomes invalid
    * 'register_kind' (integer): Register numbering scheme identifier
    * 'decl_file' (string): Source file where variable is declared (optional)
    * 'decl_line' (integer): Line number where variable is declared (optional)
    * 'type_name' (string): Type name of the variable (optional)

    Args:
        inst: SBInstruction object to annotate

    Returns:
        SBStructuredData containing variable annotation array, or invalid
        SBStructuredData if no annotations are available

    Example usage::

        annotations = annotator.AnnotateStructured(instruction)
        if annotations.IsValid():
            for i in range(annotations.GetSize()):
                item = annotations.GetItemAtIndex(i)
                name = item.GetValueForKey('variable_name').GetStringValue(1024)
                location = item.GetValueForKey('location_description').GetStringValue(1024)
                print(f'{name} -> {location}')")
lldb::SBVariableAnnotator::AnnotateStructured;