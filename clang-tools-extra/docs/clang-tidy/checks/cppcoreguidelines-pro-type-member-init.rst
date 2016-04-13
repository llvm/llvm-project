.. title:: clang-tidy - cppcoreguidelines-pro-type-member-init

cppcoreguidelines-pro-type-member-init
======================================

The check flags user-defined constructor definitions that do not
initialize all fields that would be left in an undefined state by
default construction, e.g. builtins, pointers and record types without
user-provided default constructors containing at least one such
type. If these fields aren't initialized, the constructor will leave
some of the memory in an undefined state.

For C++11 it suggests fixes to add in-class field initializers. For
older versions it inserts the field initializers into the constructor
initializer list. It will also initialize any direct base classes that
need to be zeroed in the constructor initializer list.

The check takes assignment of fields in the constructor body into
account but generates false positives for fields initialized in
methods invoked in the constructor body.

The check also flags variables of record types without a user-provided
constructor that are not initialized. The suggested fix is to zero
initialize the variable via {} for C++11 and beyond or = {} for older
versions.

IgnoreArrays option
-------------------

For performance critical code, it may be important to not zero
fixed-size array members. If on, IgnoreArrays will not warn about
array members that are not zero-initialized during construction.
IgnoreArrays is false by default.

This rule is part of the "Type safety" profile of the C++ Core
Guidelines, corresponding to rule Type.6. See
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Pro-type-memberinit.
