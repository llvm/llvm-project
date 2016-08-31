.. title:: clang-tidy - cppcoreguidelines-special-member-functions

cppcoreguidelines-special-member-functions
==========================================

The check finds classes where some but not all of the special member functions
are defined.

By default the compiler defines a copy constructor, copy assignment operator,
move constructor, move assignment operator and destructor. The default can be
suppressed by explicit user-definitions. The relationship between which
functions will be suppressed by definitions of other functions is complicated
and it is advised that all five are defaulted or explicitly defined.

Note that defining a function with ``= delete`` is considered to be a
definition.

This rule is part of the "Constructors, assignments, and destructors" profile of the C++ Core
Guidelines, corresponding to rule C.21. See

https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#c21-if-you-define-or-delete-any-default-operation-define-or-delete-them-all.
