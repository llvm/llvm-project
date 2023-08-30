STRING_EXTENSION_OUTSIDE(SBValue)
%extend lldb::SBValue {
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
                if isinstance(key, int):
                    count = len(self)
                    if -count <= key < count:
                        key %= count
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
}
