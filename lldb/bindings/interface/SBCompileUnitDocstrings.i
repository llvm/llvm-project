%feature("docstring",
"Represents a compilation unit, or compiled source file.

SBCompileUnit supports line entry iteration. For example,::

    # Now get the SBSymbolContext from this frame.  We want everything. :-)
    context = frame0.GetSymbolContext(lldb.eSymbolContextEverything)
    ...

    compileUnit = context.GetCompileUnit()

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

See also :py:class:`SBSymbolContext` and :py:class:`SBLineEntry`"
) lldb::SBCompileUnit;

%feature("docstring", "
     Get the index for a provided line entry in this compile unit.

     @param[in] line_entry
        The SBLineEntry object for which we are looking for the index.

     @param[in] exact
        An optional boolean defaulting to false that ensures that the provided
        line entry has a perfect match in the compile unit.

     @return
        The index of the user-provided line entry. UINT32_MAX if the line entry
        was not found in the compile unit.") lldb::SBCompileUnit::FindLineEntryIndex;

%feature("docstring", "
     Get all types matching type_mask from debug info in this
     compile unit.

     @param[in] type_mask
        A bitfield that consists of one or more bits logically OR'ed
        together from the lldb::TypeClass enumeration. This allows
        you to request only structure types, or only class, struct
        and union types. Passing in lldb::eTypeClassAny will return
        all types found in the debug information for this compile
        unit.

     @return
        A list of types in this compile unit that match type_mask"
) lldb::SBCompileUnit::GetTypes;
