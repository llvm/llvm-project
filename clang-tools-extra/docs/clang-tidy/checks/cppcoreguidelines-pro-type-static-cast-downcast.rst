.. title:: clang-tidy - cppcoreguidelines-pro-type-static-cast-downcast

cppcoreguidelines-pro-type-static-cast-downcast
===============================================

This check flags all usages of ``static_cast``, where a base class is casted to
a derived class. In those cases, a fixit is provided to convert the cast to a
``dynamic_cast``.

Use of these casts can violate type safety and cause the program to access a
variable that is actually of type ``X`` to be accessed as if it were of an
unrelated type ``Z``.

This rule is part of the "Type safety" profile of the C++ Core Guidelines, see
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Pro-type-downcast.
