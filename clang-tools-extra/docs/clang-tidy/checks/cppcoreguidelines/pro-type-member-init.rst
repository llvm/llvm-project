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

The check also flags variables with automatic storage duration that have record
types without a user-provided constructor and are not initialized. The suggested
fix is to zero initialize the variable via ``{}`` for C++11 and beyond or ``=
{}`` for older language versions.

Options
-------

.. option:: IgnoreArrays

   If set to `true`, the check will not warn about array members that are not
   zero-initialized during construction. For performance critical code, it may
   be important to not initialize fixed-size array members. Default is `false`.

.. option:: UseAssignment

   If set to `true`, the check will provide fix-its with literal initializers
   \( ``int i = 0;`` \) instead of curly braces \( ``int i{};`` \).

This rule is part of the `Type safety (Type.6)
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Pro-type-memberinit>`_
profile from the C++ Core Guidelines.
