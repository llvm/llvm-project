.. title:: clang-tidy - cppcoreguidelines-pro-type-member-init

cppcoreguidelines-pro-type-member-init
======================================

The check flags user-defined constructor definitions that do not initialize all
builtin and pointer fields which leaves their memory in an undefined state.

For C++11 it suggests fixes to add in-class field initializers. For older
versions it inserts the field initializers into the constructor initializer
list.

The check takes assignment of fields in the constructor body into account but
generates false positives for fields initialized in methods invoked in the
constructor body.

This rule is part of the "Type safety" profile of the C++ Core Guidelines, see
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Pro-type-memberinit.
