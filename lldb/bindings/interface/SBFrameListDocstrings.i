%feature("docstring",
"Represents a list of :py:class:`SBFrame` objects."
) lldb::SBFrameList;

%feature("autodoc", "GetSize(SBFrameList self) -> uint32_t") lldb::SBFrameList::GetSize;
%feature("docstring", "
    Returns the number of frames in the list."
) lldb::SBFrameList::GetSize;

%feature("autodoc", "GetFrameAtIndex(SBFrameList self, uint32_t idx) -> SBFrame") lldb::SBFrameList::GetFrameAtIndex;
%feature("docstring", "
    Returns the frame at the given index."
) lldb::SBFrameList::GetFrameAtIndex;