STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeMember, lldb::eDescriptionLevelBrief)
%extend lldb::SBTypeMember {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __eq__(self, other):
            return not self.__ne__(other)

        def __int__(self):
            pass

        def __hex__(self):
            pass

        def __oct__(self):
            pass

        def __len__(self):
            pass

        def __iter__(self):
            pass

        name = property(GetName, None, doc='''A read only property that returns the name for this member as a string.''')
        type = property(GetType, None, doc='''A read only property that returns an lldb object that represents the type (lldb.SBType) for this member.''')
        byte_offset = property(GetOffsetInBytes, None, doc='''A read only property that returns offset in bytes for this member as an integer.''')
        bit_offset = property(GetOffsetInBits, None, doc='''A read only property that returns offset in bits for this member as an integer.''')
        is_bitfield = property(IsBitfield, None, doc='''A read only property that returns true if this member is a bitfield.''')
        bitfield_bit_size = property(GetBitfieldSizeInBits, None, doc='''A read only property that returns the bitfield size in bits for this member as an integer, or zero if this member is not a bitfield.''')
    %}
#endif
}

STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeMemberFunction, lldb::eDescriptionLevelBrief)

%extend lldb::SBTypeMemberFunction {
#ifdef SWIGPYTHON
    // operator== is a free function, which swig does not handle, so we inject
    // our own equality operator here
    %pythoncode%{
      def __eq__(self, other):
        return not self.__ne__(other)

      def __int__(self):
        pass

      def __hex__(self):
        pass

      def __oct__(self):
        pass

      def __len__(self):
        pass

      def __iter__(self):
        pass
    %}
#endif
}

STRING_EXTENSION_LEVEL_OUTSIDE(SBType, lldb::eDescriptionLevelBrief)

%extend lldb::SBType {
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

        def __eq__(self, other):
            return not self.__ne__(other)

        def __int__(self):
            pass

        def __hex__(self):
            pass

        def __oct__(self):
            pass

        def __len__(self):
            return self.GetByteSize()

        def __iter__(self):
            pass

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
}

%extend lldb::SBTypeList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __eq__(self, other):
        return not self.__ne__(other)

    def __int__(self):
        pass

    def __hex__(self):
        pass

    def __oct__(self):
        pass

    def __iter__(self):
        '''Iterate over all types in a lldb.SBTypeList object.'''
        return lldb_iter(self, 'GetSize', 'GetTypeAtIndex')

    def __len__(self):
        '''Return the number of types in a lldb.SBTypeList object.'''
        return self.GetSize()
    %}
#endif
}
