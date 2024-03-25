%feature("docstring",
"Represents a generic function, which can be inlined or not.

For example (from test/lldbutil.py, but slightly modified for doc purpose),::

        ...

        frame = thread.GetFrameAtIndex(i)
        addr = frame.GetPCAddress()
        load_addr = addr.GetLoadAddress(target)
        function = frame.GetFunction()
        mod_name = frame.GetModule().GetFileSpec().GetFilename()

        if not function:
            # No debug info for 'function'.
            symbol = frame.GetSymbol()
            file_addr = addr.GetFileAddress()
            start_addr = symbol.GetStartAddress().GetFileAddress()
            symbol_name = symbol.GetName()
            symbol_offset = file_addr - start_addr
            print >> output, '  frame #{num}: {addr:#016x} {mod}`{symbol} + {offset}'.format(
                num=i, addr=load_addr, mod=mod_name, symbol=symbol_name, offset=symbol_offset)
        else:
            # Debug info is available for 'function'.
            func_name = frame.GetFunctionName()
            file_name = frame.GetLineEntry().GetFileSpec().GetFilename()
            line_num = frame.GetLineEntry().GetLine()
            print >> output, '  frame #{num}: {addr:#016x} {mod}`{func} at {file}:{line} {args}'.format(
                num=i, addr=load_addr, mod=mod_name,
                func='%s [inlined]' % func_name] if frame.IsInlined() else func_name,
                file=file_name, line=line_num, args=get_args_as_string(frame, showFuncName=False))

        ..."
) lldb::SBFunction;

%feature("docstring", "
    Returns true if the function was compiled with optimization.
    Optimization, in this case, is meant to indicate that the debugger
    experience may be confusing for the user -- variables optimized away,
    stepping jumping between source lines -- and the driver may want to
    provide some guidance to the user about this.
    Returns false if unoptimized, or unknown."
) lldb::SBFunction::GetIsOptimized;
