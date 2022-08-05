.. title:: clang-tidy - cppcoreguidelines-pro-type-reinterpret-cast

cppcoreguidelines-pro-type-reinterpret-cast
===========================================

This check flags all uses of ``reinterpret_cast`` in C++ code.

Use of these casts can violate type safety and cause the program to access a
variable that is actually of type ``X`` to be accessed as if it were of an
unrelated type ``Z``.

This rule is part of the "Type safety" profile of the C++ Core Guidelines, see
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Pro-type-reinterpretcast.
