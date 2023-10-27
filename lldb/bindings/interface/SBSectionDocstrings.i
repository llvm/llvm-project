%feature("docstring",
"Represents an executable image section.

SBSection supports iteration through its subsection, represented as SBSection
as well.  For example, ::

    for sec in exe_module:
        if sec.GetName() == '__TEXT':
            print sec
            break
    print INDENT + 'Number of subsections: %d' % sec.GetNumSubSections()
    for subsec in sec:
        print INDENT + repr(subsec)

produces: ::

  [0x0000000100000000-0x0000000100002000) a.out.__TEXT
      Number of subsections: 6
      [0x0000000100001780-0x0000000100001d5c) a.out.__TEXT.__text
      [0x0000000100001d5c-0x0000000100001da4) a.out.__TEXT.__stubs
      [0x0000000100001da4-0x0000000100001e2c) a.out.__TEXT.__stub_helper
      [0x0000000100001e2c-0x0000000100001f10) a.out.__TEXT.__cstring
      [0x0000000100001f10-0x0000000100001f68) a.out.__TEXT.__unwind_info
      [0x0000000100001f68-0x0000000100001ff8) a.out.__TEXT.__eh_frame

See also :py:class:`SBModule` ."
) lldb::SBSection;

%feature("docstring", "
    Return the size of a target's byte represented by this section
    in numbers of host bytes. Note that certain architectures have
    varying minimum addressable unit (i.e. byte) size for their
    CODE or DATA buses.

    @return
        The number of host (8-bit) bytes needed to hold a target byte"
) lldb::SBSection::GetTargetByteSize;
