//===-- SWIG Interface for SBValue ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents the value of a variable, a register, or an expression.

SBValue supports iteration through its child, which in turn is represented
as an SBValue.  For example, we can get the general purpose registers of a
frame as an SBValue, and iterate through all the registers,

    registerSet = frame.registers # Returns an SBValueList.
    for regs in registerSet:
        if 'general purpose registers' in regs.name.lower():
            GPRs = regs
            break

    print('%s (number of children = %d):' % (GPRs.name, GPRs.num_children))
    for reg in GPRs:
        print('Name: ', reg.name, ' Value: ', reg.value)

produces the output:

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
) SBValue;
class SBValue
{
public:
    SBValue ();

    SBValue (const SBValue &rhs);

    ~SBValue ();

    bool
    IsValid();

    explicit operator bool() const;

    void
    Clear();

    SBError
    GetError();

    lldb::user_id_t
    GetID ();

    const char *
    GetName();

    const char *
    GetTypeName ();

    const char *
    GetDisplayTypeName ();

    size_t
    GetByteSize ();

    bool
    IsInScope ();

    lldb::Format
    GetFormat ();

    void
    SetFormat (lldb::Format format);

    const char *
    GetValue ();

    int64_t
    GetValueAsSigned(SBError& error, int64_t fail_value=0);

    uint64_t
    GetValueAsUnsigned(SBError& error, uint64_t fail_value=0);

    int64_t
    GetValueAsSigned(int64_t fail_value=0);

    uint64_t
    GetValueAsUnsigned(uint64_t fail_value=0);

    ValueType
    GetValueType ();

    bool
    GetValueDidChange ();

    const char *
    GetSummary ();

    const char *
    GetSummary (lldb::SBStream& stream,
                lldb::SBTypeSummaryOptions& options);

    const char *
    GetObjectDescription ();

    lldb::SBValue
    GetDynamicValue (lldb::DynamicValueType use_dynamic);

    lldb::SBValue
    GetStaticValue ();

    lldb::SBValue
    GetNonSyntheticValue ();

    lldb::DynamicValueType
    GetPreferDynamicValue ();

    void
    SetPreferDynamicValue (lldb::DynamicValueType use_dynamic);

    bool
    GetPreferSyntheticValue ();

    void
    SetPreferSyntheticValue (bool use_synthetic);

    bool
    IsDynamic();

    bool
    IsSynthetic ();

    bool
    IsSyntheticChildrenGenerated ();

    void
    SetSyntheticChildrenGenerated (bool);

    const char *
    GetLocation ();

    bool
    SetValueFromCString (const char *value_str);

    bool
    SetValueFromCString (const char *value_str, lldb::SBError& error);

    lldb::SBTypeFormat
    GetTypeFormat ();

    lldb::SBTypeSummary
    GetTypeSummary ();

    lldb::SBTypeFilter
    GetTypeFilter ();

    lldb::SBTypeSynthetic
    GetTypeSynthetic ();

    lldb::SBValue
    GetChildAtIndex (uint32_t idx);

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
        A new SBValue object that represents the child member value.") GetChildAtIndex;
    lldb::SBValue
    GetChildAtIndex (uint32_t idx,
                     lldb::DynamicValueType use_dynamic,
                     bool can_create_synthetic);

    lldb::SBValue
    CreateChildAtOffset (const char *name, uint32_t offset, lldb::SBType type);

    lldb::SBValue
    SBValue::Cast (lldb::SBType type);

    lldb::SBValue
    CreateValueFromExpression (const char *name, const char* expression);

    lldb::SBValue
    CreateValueFromExpression (const char *name, const char* expression, SBExpressionOptions &options);

    lldb::SBValue
    CreateValueFromAddress(const char* name, lldb::addr_t address, lldb::SBType type);

  lldb::SBValue
  CreateValueFromData (const char* name,
                       lldb::SBData data,
                       lldb::SBType type);

    lldb::SBType
    GetType();

    %feature("docstring", "
    Returns the child member index.

    Matches children of this object only and will match base classes and
    member names if this is a clang typed object.

    @param[in] name
        The name of the child value to get

    @return
        An index to the child member value.") GetIndexOfChildWithName;
    uint32_t
    GetIndexOfChildWithName (const char *name);

    lldb::SBValue
    GetChildMemberWithName (const char *name);

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
        A new SBValue object that represents the child member value.") GetChildMemberWithName;
    lldb::SBValue
    GetChildMemberWithName (const char *name, lldb::DynamicValueType use_dynamic);

    %feature("docstring", "Expands nested expressions like .a->b[0].c[1]->d."
    ) GetValueForExpressionPath;
    lldb::SBValue
    GetValueForExpressionPath(const char* expr_path);

    lldb::SBDeclaration
    GetDeclaration ();

    bool
    MightHaveChildren ();

    bool
    IsRuntimeSupportValue ();

    uint32_t
    GetNumChildren ();

    %feature("doctstring", "
    Returns the number for children.

    @param[in] max
        If max is less the lldb.UINT32_MAX, then the returned value is
        capped to max.

    @return
        An integer value capped to the argument max.") GetNumChildren;
    uint32_t
    GetNumChildren (uint32_t max);

    void *
    GetOpaqueType();

    lldb::SBValue
    Dereference ();

    lldb::SBValue
    AddressOf();

    bool
    TypeIsPointerType ();

    lldb::SBTarget
    GetTarget();

    lldb::SBProcess
    GetProcess();

    lldb::SBThread
    GetThread();

    lldb::SBFrame
    GetFrame();

    %feature("docstring", "
    Find and watch a variable.
    It returns an SBWatchpoint, which may be invalid.") Watch;
    lldb::SBWatchpoint
    Watch (bool resolve_location, bool read, bool write, SBError &error);

    %feature("docstring", "
    Find and watch the location pointed to by a variable.
    It returns an SBWatchpoint, which may be invalid.") WatchPointee;
    lldb::SBWatchpoint
    WatchPointee (bool resolve_location, bool read, bool write, SBError &error);

    bool
    GetDescription (lldb::SBStream &description);

    bool
    GetExpressionPath (lldb::SBStream &description);

  %feature("docstring", "
    Get an SBData wrapping what this SBValue points to.

    This method will dereference the current SBValue, if its
    data type is a T* or T[], and extract item_count elements
    of type T from it, copying their contents in an SBData.

    @param[in] item_idx
        The index of the first item to retrieve. For an array
        this is equivalent to array[item_idx], for a pointer
        to *(pointer + item_idx). In either case, the measurement
        unit for item_idx is the sizeof(T) rather than the byte

    @param[in] item_count
        How many items should be copied into the output. By default
        only one item is copied, but more can be asked for.

    @return
        An SBData with the contents of the copied items, on success.
        An empty SBData otherwise.") GetPointeeData;
  lldb::SBData
  GetPointeeData (uint32_t item_idx = 0,
          uint32_t item_count = 1);

    %feature("docstring", "
    Get an SBData wrapping the contents of this SBValue.

    This method will read the contents of this object in memory
    and copy them into an SBData for future use.

    @return
        An SBData with the contents of this SBValue, on success.
        An empty SBData otherwise.") GetData;
    lldb::SBData
    GetData ();

    bool
    SetData (lldb::SBData &data, lldb::SBError& error);

  lldb::addr_t
  GetLoadAddress();

  lldb::SBAddress
  GetAddress();

    lldb::SBValue
    Persist ();

    %feature("docstring", "Returns an expression path for this value."
    ) GetExpressionPath;
    bool
    GetExpressionPath (lldb::SBStream &description, bool qualify_cxx_base_classes);

    lldb::SBValue
    EvaluateExpression(const char *expr) const;

    lldb::SBValue
    EvaluateExpression(const char *expr,
                       const SBExpressionOptions &options) const;

    lldb::SBValue
    EvaluateExpression(const char *expr,
                       const SBExpressionOptions &options,
                       const char *name) const;

#ifdef SWIGPYTHON
    %pythoncode %{
        def __get_dynamic__ (self):
            '''Helper function for the "SBValue.dynamic" property.'''
            return self.GetDynamicValue (eDynamicCanRunTarget)

        class children_access(object):
            '''A helper object that will lazily hand out thread for a process when supplied an index.'''

            def __init__(self, sbvalue):
                self.sbvalue = sbvalue

            def __len__(self):
                if self.sbvalue:
                    return int(self.sbvalue.GetNumChildren())
                return 0

            def __getitem__(self, key):
                if type(key) is int and key < len(self):
                    return self.sbvalue.GetChildAtIndex(key)
                return None

        def get_child_access_object(self):
            '''An accessor function that returns a children_access() object which allows lazy member variable access from a lldb.SBValue object.'''
            return self.children_access (self)

        def get_value_child_list(self):
            '''An accessor function that returns a list() that contains all children in a lldb.SBValue object.'''
            children = []
            accessor = self.get_child_access_object()
            for idx in range(len(accessor)):
                children.append(accessor[idx])
            return children

        def __iter__(self):
            '''Iterate over all child values of a lldb.SBValue object.'''
            return lldb_iter(self, 'GetNumChildren', 'GetChildAtIndex')

        def __len__(self):
            '''Return the number of child values of a lldb.SBValue object.'''
            return self.GetNumChildren()

        children = property(get_value_child_list, None, doc='''A read only property that returns a list() of lldb.SBValue objects for the children of the value.''')
        child = property(get_child_access_object, None, doc='''A read only property that returns an object that can access children of a variable by index (child_value = value.children[12]).''')
        name = property(GetName, None, doc='''A read only property that returns the name of this value as a string.''')
        type = property(GetType, None, doc='''A read only property that returns a lldb.SBType object that represents the type for this value.''')
        size = property(GetByteSize, None, doc='''A read only property that returns the size in bytes of this value.''')
        is_in_scope = property(IsInScope, None, doc='''A read only property that returns a boolean value that indicates whether this value is currently lexically in scope.''')
        format = property(GetName, SetFormat, doc='''A read/write property that gets/sets the format used for lldb.SBValue().GetValue() for this value. See enumerations that start with "lldb.eFormat".''')
        value = property(GetValue, SetValueFromCString, doc='''A read/write property that gets/sets value from a string.''')
        value_type = property(GetValueType, None, doc='''A read only property that returns an lldb enumeration value (see enumerations that start with "lldb.eValueType") that represents the type of this value (local, argument, global, register, etc.).''')
        changed = property(GetValueDidChange, None, doc='''A read only property that returns a boolean value that indicates if this value has changed since it was last updated.''')
        data = property(GetData, None, doc='''A read only property that returns an lldb object (lldb.SBData) that represents the bytes that make up the value for this object.''')
        load_addr = property(GetLoadAddress, None, doc='''A read only property that returns the load address of this value as an integer.''')
        addr = property(GetAddress, None, doc='''A read only property that returns an lldb.SBAddress that represents the address of this value if it is in memory.''')
        deref = property(Dereference, None, doc='''A read only property that returns an lldb.SBValue that is created by dereferencing this value.''')
        address_of = property(AddressOf, None, doc='''A read only property that returns an lldb.SBValue that represents the address-of this value.''')
        error = property(GetError, None, doc='''A read only property that returns the lldb.SBError that represents the error from the last time the variable value was calculated.''')
        summary = property(GetSummary, None, doc='''A read only property that returns the summary for this value as a string''')
        description = property(GetObjectDescription, None, doc='''A read only property that returns the language-specific description of this value as a string''')
        dynamic = property(__get_dynamic__, None, doc='''A read only property that returns an lldb.SBValue that is created by finding the dynamic type of this value.''')
        location = property(GetLocation, None, doc='''A read only property that returns the location of this value as a string.''')
        target = property(GetTarget, None, doc='''A read only property that returns the lldb.SBTarget that this value is associated with.''')
        process = property(GetProcess, None, doc='''A read only property that returns the lldb.SBProcess that this value is associated with, the returned value might be invalid and should be tested.''')
        thread = property(GetThread, None, doc='''A read only property that returns the lldb.SBThread that this value is associated with, the returned value might be invalid and should be tested.''')
        frame = property(GetFrame, None, doc='''A read only property that returns the lldb.SBFrame that this value is associated with, the returned value might be invalid and should be tested.''')
        num_children = property(GetNumChildren, None, doc='''A read only property that returns the number of child lldb.SBValues that this value has.''')
        unsigned = property(GetValueAsUnsigned, None, doc='''A read only property that returns the value of this SBValue as an usigned integer.''')
        signed = property(GetValueAsSigned, None, doc='''A read only property that returns the value of this SBValue as a signed integer.''')

        def get_expr_path(self):
            s = SBStream()
            self.GetExpressionPath (s)
            return s.GetData()

        path = property(get_expr_path, None, doc='''A read only property that returns the expression path that one can use to reach this value in an expression.''')

        def synthetic_child_from_expression(self, name, expr, options=None):
            if options is None: options = lldb.SBExpressionOptions()
            child = self.CreateValueFromExpression(name, expr, options)
            child.SetSyntheticChildrenGenerated(True)
            return child

        def synthetic_child_from_data(self, name, data, type):
            child = self.CreateValueFromData(name, data, type)
            child.SetSyntheticChildrenGenerated(True)
            return child

        def synthetic_child_from_address(self, name, addr, type):
            child = self.CreateValueFromAddress(name, addr, type)
            child.SetSyntheticChildrenGenerated(True)
            return child

        def __eol_test(val):
            """Default function for end of list test takes an SBValue object.

            Return True if val is invalid or it corresponds to a null pointer.
            Otherwise, return False.
            """
            if not val or val.GetValueAsUnsigned() == 0:
                return True
            else:
                return False

        # ==================================================
        # Iterator for lldb.SBValue treated as a linked list
        # ==================================================
        def linked_list_iter(self, next_item_name, end_of_list_test=__eol_test):
            """Generator adaptor to support iteration for SBValue as a linked list.

            linked_list_iter() is a special purpose iterator to treat the SBValue as
            the head of a list data structure, where you specify the child member
            name which points to the next item on the list and you specify the
            end-of-list test function which takes an SBValue for an item and returns
            True if EOL is reached and False if not.

            linked_list_iter() also detects infinite loop and bails out early.

            The end_of_list_test arg, if omitted, defaults to the __eol_test
            function above.

            For example,

            # Get Frame #0.
            ...

            # Get variable 'task_head'.
            task_head = frame0.FindVariable('task_head')
            ...

            for t in task_head.linked_list_iter('next'):
                print t
            """
            if end_of_list_test(self):
                return
            item = self
            visited = set()
            try:
                while not end_of_list_test(item) and not item.GetValueAsUnsigned() in visited:
                    visited.add(item.GetValueAsUnsigned())
                    yield item
                    # Prepare for the next iteration.
                    item = item.GetChildMemberWithName(next_item_name)
            except:
                # Exception occurred.  Stop the generator.
                pass

            return
    %}
#endif

};

} // namespace lldb
