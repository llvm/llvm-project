%feature("docstring",
"Represents a lexical block. SBFunction contains SBBlock(s)."
) lldb::SBBlock;

%feature("docstring",
"Is this block contained within an inlined function?"
) lldb::SBBlock::IsInlined;

%feature("docstring", "
    Get the function name if this block represents an inlined function;
    otherwise, return None.") lldb::SBBlock::GetInlinedName;

%feature("docstring", "
    Get the call site file if this block represents an inlined function;
    otherwise, return an invalid file spec.") lldb::SBBlock::GetInlinedCallSiteFile;

%feature("docstring", "
    Get the call site line if this block represents an inlined function;
    otherwise, return 0.") lldb::SBBlock::GetInlinedCallSiteLine;

%feature("docstring", "
    Get the call site column if this block represents an inlined function;
    otherwise, return 0.") lldb::SBBlock::GetInlinedCallSiteColumn;

%feature("docstring", "Get the parent block.") lldb::SBBlock::GetParent;

%feature("docstring", "Get the inlined block that is or contains this block."
) lldb::SBBlock::GetContainingInlinedBlock;

%feature("docstring", "Get the sibling block for this block.") lldb::SBBlock::GetSibling;

%feature("docstring", "Get the first child block.") lldb::SBBlock::GetFirstChild;
