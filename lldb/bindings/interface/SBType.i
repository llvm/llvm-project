//===-- SWIG Interface for SBType -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

    %feature("docstring",
"Represents a member of a type.") SBTypeMember;

class SBTypeMember
{
public:
    SBTypeMember ();

    SBTypeMember (const lldb::SBTypeMember& rhs);

    ~SBTypeMember();

    bool
    IsValid() const;

    explicit operator bool() const;

    const char *
    GetName ();

    lldb::SBType
    GetType ();

    uint64_t
    GetOffsetInBytes();

    uint64_t
    GetOffsetInBits();

    bool
    IsBitfield();

    uint32_t
    GetBitfieldSizeInBits();

    STRING_EXTENSION_LEVEL(SBTypeMember, lldb::eDescriptionLevelBrief)

#ifdef SWIGPYTHON
    %pythoncode %{
        name = property(GetName, None, doc='''A read only property that returns the name for this member as a string.''')
        type = property(GetType, None, doc='''A read only property that returns an lldb object that represents the type (lldb.SBType) for this member.''')
        byte_offset = property(GetOffsetInBytes, None, doc='''A read only property that returns offset in bytes for this member as an integer.''')
        bit_offset = property(GetOffsetInBits, None, doc='''A read only property that returns offset in bits for this member as an integer.''')
        is_bitfield = property(IsBitfield, None, doc='''A read only property that returns true if this member is a bitfield.''')
        bitfield_bit_size = property(GetBitfieldSizeInBits, None, doc='''A read only property that returns the bitfield size in bits for this member as an integer, or zero if this member is not a bitfield.''')
    %}
#endif

protected:
    std::unique_ptr<lldb_private::TypeMemberImpl> m_opaque_ap;
};

%feature("docstring",
"Represents a member function of a type."
) SBTypeMemberFunction;
class SBTypeMemberFunction
{
public:
    SBTypeMemberFunction ();

    SBTypeMemberFunction (const lldb::SBTypeMemberFunction& rhs);

    ~SBTypeMemberFunction();

    bool
    IsValid() const;

    explicit operator bool() const;

    const char *
    GetName ();

    const char *
    GetDemangledName ();

    const char *
    GetMangledName ();

    lldb::SBType
    GetType ();

    lldb::SBType
    GetReturnType ();

    uint32_t
    GetNumberOfArguments ();

    lldb::SBType
    GetArgumentTypeAtIndex (uint32_t);

    lldb::MemberFunctionKind
    GetKind();

    bool
    GetDescription (lldb::SBStream &description,
                    lldb::DescriptionLevel description_level);

    STRING_EXTENSION_LEVEL(SBTypeMemberFunction, lldb::eDescriptionLevelBrief)
protected:
    lldb::TypeMemberFunctionImplSP m_opaque_sp;
};

%feature("docstring",
"Represents a data type in lldb.  The FindFirstType() method of SBTarget/SBModule
returns a SBType.

SBType supports the eq/ne operator. For example,::

    //main.cpp:

    class Task {
    public:
        int id;
        Task *next;
        Task(int i, Task *n):
            id(i),
            next(n)
        {}
    };

    int main (int argc, char const *argv[])
    {
        Task *task_head = new Task(-1, NULL);
        Task *task1 = new Task(1, NULL);
        Task *task2 = new Task(2, NULL);
        Task *task3 = new Task(3, NULL); // Orphaned.
        Task *task4 = new Task(4, NULL);
        Task *task5 = new Task(5, NULL);

        task_head->next = task1;
        task1->next = task2;
        task2->next = task4;
        task4->next = task5;

        int total = 0;
        Task *t = task_head;
        while (t != NULL) {
            if (t->id >= 0)
                ++total;
            t = t->next;
        }
        printf('We have a total number of %d tasks\\n', total);

        // This corresponds to an empty task list.
        Task *empty_task_head = new Task(-1, NULL);

        return 0; // Break at this line
    }

    # find_type.py:

            # Get the type 'Task'.
            task_type = target.FindFirstType('Task')
            self.assertTrue(task_type)

            # Get the variable 'task_head'.
            frame0.FindVariable('task_head')
            task_head_type = task_head.GetType()
            self.assertTrue(task_head_type.IsPointerType())

            # task_head_type is 'Task *'.
            task_pointer_type = task_type.GetPointerType()
            self.assertTrue(task_head_type == task_pointer_type)

            # Get the child mmember 'id' from 'task_head'.
            id = task_head.GetChildMemberWithName('id')
            id_type = id.GetType()

            # SBType.GetBasicType() takes an enum 'BasicType' (lldb-enumerations.h).
            int_type = id_type.GetBasicType(lldb.eBasicTypeInt)
            # id_type and int_type should be the same type!
            self.assertTrue(id_type == int_type)

...") SBType;
class SBType
{
public:
    SBType ();

    SBType (const lldb::SBType &rhs);

    ~SBType ();

    bool
    IsValid();

    explicit operator bool() const;

    uint64_t
    GetByteSize();

    bool
    IsPointerType();

    bool
    IsReferenceType();

    bool
    IsFunctionType ();

    bool
    IsPolymorphicClass ();

    bool
    IsArrayType ();

    bool
    IsVectorType ();

    bool
    IsTypedefType ();

    bool
    IsAnonymousType ();

    bool
    IsScopedEnumerationType ();

    lldb::SBType
    GetPointerType();

    lldb::SBType
    GetPointeeType();

    lldb::SBType
    GetReferenceType();

    lldb::SBType
    SBType::GetTypedefedType();

    lldb::SBType
    GetDereferencedType();

    lldb::SBType
    GetUnqualifiedType();

    lldb::SBType
    GetCanonicalType();

    lldb::SBType
    GetEnumerationIntegerType();

    lldb::SBType
    GetArrayElementType ();

    lldb::SBType
    GetArrayType (uint64_t size);

    lldb::SBType
    GetVectorElementType ();

    lldb::BasicType
    GetBasicType();

    lldb::SBType
    GetBasicType (lldb::BasicType type);

    uint32_t
    GetNumberOfFields ();

    uint32_t
    GetNumberOfDirectBaseClasses ();

    uint32_t
    GetNumberOfVirtualBaseClasses ();

    lldb::SBTypeMember
    GetFieldAtIndex (uint32_t idx);

    lldb::SBTypeMember
    GetDirectBaseClassAtIndex (uint32_t idx);

    lldb::SBTypeMember
    GetVirtualBaseClassAtIndex (uint32_t idx);

    lldb::SBTypeEnumMemberList
    GetEnumMembers();

    lldb::SBModule
    GetModule();

    const char*
    GetName();

    const char *
    GetDisplayTypeName ();

    lldb::TypeClass
    GetTypeClass ();

    uint32_t
    GetNumberOfTemplateArguments ();

    lldb::SBType
    GetTemplateArgumentType (uint32_t idx);

    lldb::TemplateArgumentKind
    GetTemplateArgumentKind (uint32_t idx);

    lldb::SBType
    GetFunctionReturnType ();

    lldb::SBTypeList
    GetFunctionArgumentTypes ();

    uint32_t
    GetNumberOfMemberFunctions ();

    lldb::SBTypeMemberFunction
    GetMemberFunctionAtIndex (uint32_t idx);

    bool
    IsTypeComplete ();

    uint32_t
    GetTypeFlags ();

    bool operator==(lldb::SBType &rhs);

    bool operator!=(lldb::SBType &rhs);

    STRING_EXTENSION_LEVEL(SBType, lldb::eDescriptionLevelBrief)

#ifdef SWIGPYTHON
    %pythoncode %{
        def template_arg_array(self):
            num_args = self.num_template_args
            if num_args:
                template_args = []
                for i in range(num_args):
                    template_args.append(self.GetTemplateArgumentType(i))
                return template_args
            return None

        module = property(GetModule, None, doc='''A read only property that returns the module in which type is defined.''')
        name = property(GetName, None, doc='''A read only property that returns the name for this type as a string.''')
        size = property(GetByteSize, None, doc='''A read only property that returns size in bytes for this type as an integer.''')
        is_pointer = property(IsPointerType, None, doc='''A read only property that returns a boolean value that indicates if this type is a pointer type.''')
        is_reference = property(IsReferenceType, None, doc='''A read only property that returns a boolean value that indicates if this type is a reference type.''')
        is_reference = property(IsReferenceType, None, doc='''A read only property that returns a boolean value that indicates if this type is a function type.''')
        num_fields = property(GetNumberOfFields, None, doc='''A read only property that returns number of fields in this type as an integer.''')
        num_bases = property(GetNumberOfDirectBaseClasses, None, doc='''A read only property that returns number of direct base classes in this type as an integer.''')
        num_vbases = property(GetNumberOfVirtualBaseClasses, None, doc='''A read only property that returns number of virtual base classes in this type as an integer.''')
        num_template_args = property(GetNumberOfTemplateArguments, None, doc='''A read only property that returns number of template arguments in this type as an integer.''')
        template_args = property(template_arg_array, None, doc='''A read only property that returns a list() of lldb.SBType objects that represent all template arguments in this type.''')
        type = property(GetTypeClass, None, doc='''A read only property that returns an lldb enumeration value (see enumerations that start with "lldb.eTypeClass") that represents a classification for this type.''')
        is_complete = property(IsTypeComplete, None, doc='''A read only property that returns a boolean value that indicates if this type is a complete type (True) or a forward declaration (False).''')

        def get_bases_array(self):
            '''An accessor function that returns a list() that contains all direct base classes in a lldb.SBType object.'''
            bases = []
            for idx in range(self.GetNumberOfDirectBaseClasses()):
                bases.append(self.GetDirectBaseClassAtIndex(idx))
            return bases

        def get_vbases_array(self):
            '''An accessor function that returns a list() that contains all fields in a lldb.SBType object.'''
            vbases = []
            for idx in range(self.GetNumberOfVirtualBaseClasses()):
                vbases.append(self.GetVirtualBaseClassAtIndex(idx))
            return vbases

        def get_fields_array(self):
            '''An accessor function that returns a list() that contains all fields in a lldb.SBType object.'''
            fields = []
            for idx in range(self.GetNumberOfFields()):
                fields.append(self.GetFieldAtIndex(idx))
            return fields

        def get_members_array(self):
            '''An accessor function that returns a list() that contains all members (base classes and fields) in a lldb.SBType object in ascending bit offset order.'''
            members = []
            bases = self.get_bases_array()
            fields = self.get_fields_array()
            vbases = self.get_vbases_array()
            for base in bases:
                bit_offset = base.bit_offset
                added = False
                for idx, member in enumerate(members):
                    if member.bit_offset > bit_offset:
                        members.insert(idx, base)
                        added = True
                        break
                if not added:
                    members.append(base)
            for vbase in vbases:
                bit_offset = vbase.bit_offset
                added = False
                for idx, member in enumerate(members):
                    if member.bit_offset > bit_offset:
                        members.insert(idx, vbase)
                        added = True
                        break
                if not added:
                    members.append(vbase)
            for field in fields:
                bit_offset = field.bit_offset
                added = False
                for idx, member in enumerate(members):
                    if member.bit_offset > bit_offset:
                        members.insert(idx, field)
                        added = True
                        break
                if not added:
                    members.append(field)
            return members

        def get_enum_members_array(self):
            '''An accessor function that returns a list() that contains all enum members in an lldb.SBType object.'''
            enum_members_list = []
            sb_enum_members = self.GetEnumMembers()
            for idx in range(sb_enum_members.GetSize()):
                enum_members_list.append(sb_enum_members.GetTypeEnumMemberAtIndex(idx))
            return enum_members_list

        bases = property(get_bases_array, None, doc='''A read only property that returns a list() of lldb.SBTypeMember objects that represent all of the direct base classes for this type.''')
        vbases = property(get_vbases_array, None, doc='''A read only property that returns a list() of lldb.SBTypeMember objects that represent all of the virtual base classes for this type.''')
        fields = property(get_fields_array, None, doc='''A read only property that returns a list() of lldb.SBTypeMember objects that represent all of the fields for this type.''')
        members = property(get_members_array, None, doc='''A read only property that returns a list() of all lldb.SBTypeMember objects that represent all of the base classes, virtual base classes and fields for this type in ascending bit offset order.''')
        enum_members = property(get_enum_members_array, None, doc='''A read only property that returns a list() of all lldb.SBTypeEnumMember objects that represent the enum members for this type.''')
        %}
#endif

};

%feature("docstring",
"Represents a list of :py:class:`SBType` s.

The FindTypes() method of :py:class:`SBTarget`/:py:class:`SBModule` returns a SBTypeList.

SBTypeList supports :py:class:`SBType` iteration. For example,

.. code-block:: cpp

    // main.cpp:

    class Task {
    public:
        int id;
        Task *next;
        Task(int i, Task *n):
            id(i),
            next(n)
        {}
    };

.. code-block:: python

    # find_type.py:

    # Get the type 'Task'.
    type_list = target.FindTypes('Task')
    self.assertTrue(len(type_list) == 1)
    # To illustrate the SBType iteration.
    for type in type_list:
        # do something with type

") SBTypeList;
class SBTypeList
{
public:
    SBTypeList();

    bool
    IsValid();

    explicit operator bool() const;

    void
    Append (lldb::SBType type);

    lldb::SBType
    GetTypeAtIndex (uint32_t index);

    uint32_t
    GetSize();

    ~SBTypeList();

#ifdef SWIGPYTHON
    %pythoncode%{
    def __iter__(self):
        '''Iterate over all types in a lldb.SBTypeList object.'''
        return lldb_iter(self, 'GetSize', 'GetTypeAtIndex')

    def __len__(self):
        '''Return the number of types in a lldb.SBTypeList object.'''
        return self.GetSize()
    %}
#endif
};

} // namespace lldb
