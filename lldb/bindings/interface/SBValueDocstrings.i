%feature("docstring",
"Represents the value of a variable, a register, or an expression.

SBValue supports iteration through its child, which in turn is represented
as an SBValue.  For example, we can get the general purpose registers of a
frame as an SBValue, and iterate through all the registers,::

    registerSet = frame.registers # Returns an SBValueList.
    for regs in registerSet:
        if 'general purpose registers' in regs.name.lower():
            GPRs = regs
            break

    print('%s (number of children = %d):' % (GPRs.name, GPRs.num_children))
    for reg in GPRs:
        print('Name: ', reg.name, ' Value: ', reg.value)

produces the output: ::

    General Purpose Registers (number of children = 21):
    Name:  rax  Value:  0x0000000100000c5c
    Name:  rbx  Value:  0x0000000000000000
    Name:  rcx  Value:  0x00007fff5fbffec0
    Name:  rdx  Value:  0x00007fff5fbffeb8
    Name:  rdi  Value:  0x0000000000000001
    Name:  rsi  Value:  0x00007fff5fbffea8
    Name:  rbp  Value:  0x00007fff5fbffe80
    Name:  rsp  Value:  0x00007fff5fbffe60
    Name:  r8  Value:  0x0000000008668682
    Name:  r9  Value:  0x0000000000000000
    Name:  r10  Value:  0x0000000000001200
    Name:  r11  Value:  0x0000000000000206
    Name:  r12  Value:  0x0000000000000000
    Name:  r13  Value:  0x0000000000000000
    Name:  r14  Value:  0x0000000000000000
    Name:  r15  Value:  0x0000000000000000
    Name:  rip  Value:  0x0000000100000dae
    Name:  rflags  Value:  0x0000000000000206
    Name:  cs  Value:  0x0000000000000027
    Name:  fs  Value:  0x0000000000000010
    Name:  gs  Value:  0x0000000000000048

See also linked_list_iter() for another perspective on how to iterate through an
SBValue instance which interprets the value object as representing the head of a
linked list."
) lldb::SBValue;

%feature("docstring", "
    Get a child value by index from a value.

    Structs, unions, classes, arrays and pointers have child
    values that can be access by index.

    Structs and unions access child members using a zero based index
    for each child member. For

    Classes reserve the first indexes for base classes that have
    members (empty base classes are omitted), and all members of the
    current class will then follow the base classes.

    Pointers differ depending on what they point to. If the pointer
    points to a simple type, the child at index zero
    is the only child value available, unless synthetic_allowed
    is true, in which case the pointer will be used as an array
    and can create 'synthetic' child values using positive or
    negative indexes. If the pointer points to an aggregate type
    (an array, class, union, struct), then the pointee is
    transparently skipped and any children are going to be the indexes
    of the child values within the aggregate type. For example if
    we have a 'Point' type and we have a SBValue that contains a
    pointer to a 'Point' type, then the child at index zero will be
    the 'x' member, and the child at index 1 will be the 'y' member
    (the child at index zero won't be a 'Point' instance).

    If you actually need an SBValue that represents the type pointed
    to by a SBValue for which GetType().IsPointeeType() returns true,
    regardless of the pointee type, you can do that with the SBValue.Dereference
    method (or the equivalent deref property).

    Arrays have a preset number of children that can be accessed by
    index and will returns invalid child values for indexes that are
    out of bounds unless the synthetic_allowed is true. In this
    case the array can create 'synthetic' child values for indexes
    that aren't in the array bounds using positive or negative
    indexes.

    @param[in] idx
        The index of the child value to get

    @param[in] use_dynamic
        An enumeration that specifies whether to get dynamic values,
        and also if the target can be run to figure out the dynamic
        type of the child value.

    @param[in] synthetic_allowed
        If true, then allow child values to be created by index
        for pointers and arrays for indexes that normally wouldn't
        be allowed.

    @return
        A new SBValue object that represents the child member value."
) lldb::SBValue::GetChildAtIndex;

%feature("docstring", "
    Returns the child member index.

    Matches children of this object only and will match base classes and
    member names if this is a clang typed object.

    @param[in] name
        The name of the child value to get

    @return
        An index to the child member value."
) lldb::SBValue::GetIndexOfChildWithName;

%feature("docstring", "
    Returns the child member value.

    Matches child members of this object and child members of any base
    classes.

    @param[in] name
        The name of the child value to get

    @param[in] use_dynamic
        An enumeration that specifies whether to get dynamic values,
        and also if the target can be run to figure out the dynamic
        type of the child value.

    @return
        A new SBValue object that represents the child member value."
) lldb::SBValue::GetChildMemberWithName;

%feature("docstring", "Expands nested expressions like .a->b[0].c[1]->d."
) lldb::SBValue::GetValueForExpressionPath;

%feature("doctstring", "
    Returns the number for children.

    @param[in] max
        If max is less the lldb.UINT32_MAX, then the returned value is
        capped to max.

    @return
        An integer value capped to the argument max."
) lldb::SBValue::GetNumChildren;

%feature("docstring", "
    Find and watch a variable.
    It returns an SBWatchpoint, which may be invalid."
) lldb::SBValue::Watch;

%feature("docstring", "
    Find and watch the location pointed to by a variable.
    It returns an SBWatchpoint, which may be invalid."
) lldb::SBValue::WatchPointee;

%feature("docstring", "
    Get an SBData wrapping what this SBValue points to.

    This method will dereference the current SBValue, if its
    data type is a ``T\*`` or ``T[]``, and extract ``item_count`` elements
    of type ``T`` from it, copying their contents in an :py:class:`SBData`.

    :param item_idx: The index of the first item to retrieve. For an array
        this is equivalent to array[item_idx], for a pointer
        to ``\*(pointer + item_idx)``. In either case, the measurement
        unit for item_idx is the ``sizeof(T)`` rather than the byte
    :param item_count: How many items should be copied into the output. By default
        only one item is copied, but more can be asked for.
    :return: The contents of the copied items on success. An empty :py:class:`SBData` otherwise.
    :rtype: SBData
    "
) lldb::SBValue::GetPointeeData;

%feature("docstring", "
    Get an SBData wrapping the contents of this SBValue.

    This method will read the contents of this object in memory
    and copy them into an SBData for future use.

    @return
        An SBData with the contents of this SBValue, on success.
        An empty SBData otherwise."
) lldb::SBValue::GetData;

%feature("docstring", "Returns an expression path for this value."
) lldb::SBValue::GetExpressionPath;
