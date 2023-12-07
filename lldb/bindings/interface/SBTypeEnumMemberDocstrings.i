%feature(
    "docstring",
    "Represents a member of an enum in lldb."
) lldb::SBTypeEnumMember;

%feature("docstring",
"Represents a list of SBTypeEnumMembers.

SBTypeEnumMemberList supports SBTypeEnumMember iteration.
It also supports [] access either by index, or by enum
element name by doing: ::

  myType = target.FindFirstType('MyEnumWithElementA')
  members = myType.GetEnumMembers()
  first_elem = members[0]
  elem_A = members['A']

") lldb::SBTypeEnumMemberList;
