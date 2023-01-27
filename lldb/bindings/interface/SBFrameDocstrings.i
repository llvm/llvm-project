%feature("docstring",
"Represents one of the stack frames associated with a thread.

SBThread contains SBFrame(s). For example (from test/lldbutil.py), ::

    def print_stacktrace(thread, string_buffer = False):
        '''Prints a simple stack trace of this thread.'''

        ...

        for i in range(depth):
            frame = thread.GetFrameAtIndex(i)
            function = frame.GetFunction()

            load_addr = addrs[i].GetLoadAddress(target)
            if not function:
                file_addr = addrs[i].GetFileAddress()
                start_addr = frame.GetSymbol().GetStartAddress().GetFileAddress()
                symbol_offset = file_addr - start_addr
                print >> output, '  frame #{num}: {addr:#016x} {mod}`{symbol} + {offset}'.format(
                    num=i, addr=load_addr, mod=mods[i], symbol=symbols[i], offset=symbol_offset)
            else:
                print >> output, '  frame #{num}: {addr:#016x} {mod}`{func} at {file}:{line} {args}'.format(
                    num=i, addr=load_addr, mod=mods[i],
                    func='%s [inlined]' % funcs[i] if frame.IsInlined() else funcs[i],
                    file=files[i], line=lines[i],
                    args=get_args_as_string(frame, showFuncName=False) if not frame.IsInlined() else '()')

        ...

And, ::

    for frame in thread:
        print frame

See also SBThread."
) lldb::SBFrame;

%feature("docstring", "
    Get the Canonical Frame Address for this stack frame.
    This is the DWARF standard's definition of a CFA, a stack address
    that remains constant throughout the lifetime of the function.
    Returns an lldb::addr_t stack address, or LLDB_INVALID_ADDRESS if
    the CFA cannot be determined."
) lldb::SBFrame::GetCFA;

%feature("docstring", "
    Gets the deepest block that contains the frame PC.

    See also GetFrameBlock()."
) lldb::SBFrame::GetBlock;

    %feature("docstring", "
    Get the appropriate function name for this frame. Inlined functions in
    LLDB are represented by Blocks that have inlined function information, so
    just looking at the SBFunction or SBSymbol for a frame isn't enough.
    This function will return the appropriate function, symbol or inlined
    function name for the frame.

    This function returns:
    - the name of the inlined function (if there is one)
    - the name of the concrete function (if there is one)
    - the name of the symbol (if there is one)
    - NULL

    See also IsInlined()."
) lldb::SBFrame::GetFunctionName;

%feature("docstring", "
    Returns the language of the frame's SBFunction, or if there.
    is no SBFunction, guess the language from the mangled name.
    ."
) lldb::SBFrame::GuessLanguage;

%feature("docstring", "
    Return true if this frame represents an inlined function.

    See also GetFunctionName()."
) lldb::SBFrame::IsInlined;

%feature("docstring", "
    Return true if this frame is artificial (e.g a frame synthesized to
    capture a tail call). Local variables may not be available in an artificial
    frame."
) lldb::SBFrame::IsArtificial;

%feature("docstring", "
    The version that doesn't supply a 'use_dynamic' value will use the
    target's default."
) lldb::SBFrame::EvaluateExpression;

%feature("docstring", "
    Gets the lexical block that defines the stack frame. Another way to think
    of this is it will return the block that contains all of the variables
    for a stack frame. Inlined functions are represented as SBBlock objects
    that have inlined function information: the name of the inlined function,
    where it was called from. The block that is returned will be the first
    block at or above the block for the PC (SBFrame::GetBlock()) that defines
    the scope of the frame. When a function contains no inlined functions,
    this will be the top most lexical block that defines the function.
    When a function has inlined functions and the PC is currently
    in one of those inlined functions, this method will return the inlined
    block that defines this frame. If the PC isn't currently in an inlined
    function, the lexical block that defines the function is returned."
) lldb::SBFrame::GetFrameBlock;

%feature("docstring", "
    The version that doesn't supply a 'use_dynamic' value will use the
    target's default."
) lldb::SBFrame::GetVariables;

%feature("docstring", "
    The version that doesn't supply a 'use_dynamic' value will use the
    target's default."
) lldb::SBFrame::FindVariable;

%feature("docstring", "
    Get a lldb.SBValue for a variable path.

    Variable paths can include access to pointer or instance members: ::

        rect_ptr->origin.y
        pt.x

    Pointer dereferences: ::

        *this->foo_ptr
        **argv

    Address of: ::

        &pt
        &my_array[3].x

    Array accesses and treating pointers as arrays: ::

        int_array[1]
        pt_ptr[22].x

    Unlike `EvaluateExpression()` which returns :py:class:`SBValue` objects
    with constant copies of the values at the time of evaluation,
    the result of this function is a value that will continue to
    track the current value of the value as execution progresses
    in the current frame."
) lldb::SBFrame::GetValueForVariablePath;

%feature("docstring", "
    Find variables, register sets, registers, or persistent variables using
    the frame as the scope.

    The version that doesn't supply a ``use_dynamic`` value will use the
    target's default."
) lldb::SBFrame::FindValue;
