.. title:: clang-tidy - cppcoreguidelines-pro-type-reinterpret-cast

cppcoreguidelines-pro-type-reinterpret-cast
===========================================

This check flags all uses of ``reinterpret_cast`` in C++ code.

Use of these casts can violate type safety and cause the program to access a
variable that is actually of type ``X`` to be accessed as if it were of an
unrelated type ``Z``.

This rule is part of the `Type safety (Type.1.1)
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Pro-type-reinterpretcast>`_
profile from the C++ Core Guidelines.
