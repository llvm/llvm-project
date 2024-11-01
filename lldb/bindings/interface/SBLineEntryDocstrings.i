%feature("docstring",
"Specifies an association with a contiguous range of instructions and
a source file location.

:py:class:`SBCompileUnit` contains SBLineEntry(s). For example, ::

    for lineEntry in compileUnit:
        print('line entry: %s:%d' % (str(lineEntry.GetFileSpec()),
                                    lineEntry.GetLine()))
        print('start addr: %s' % str(lineEntry.GetStartAddress()))
        print('end   addr: %s' % str(lineEntry.GetEndAddress()))

produces: ::

    line entry: /Volumes/data/lldb/svn/trunk/test/python_api/symbol-context/main.c:20
    start addr: a.out[0x100000d98]
    end   addr: a.out[0x100000da3]
    line entry: /Volumes/data/lldb/svn/trunk/test/python_api/symbol-context/main.c:21
    start addr: a.out[0x100000da3]
    end   addr: a.out[0x100000da9]
    line entry: /Volumes/data/lldb/svn/trunk/test/python_api/symbol-context/main.c:22
    start addr: a.out[0x100000da9]
    end   addr: a.out[0x100000db6]
    line entry: /Volumes/data/lldb/svn/trunk/test/python_api/symbol-context/main.c:23
    start addr: a.out[0x100000db6]
    end   addr: a.out[0x100000dbc]
    ...

See also :py:class:`SBCompileUnit` ."
) lldb::SBLineEntry;
