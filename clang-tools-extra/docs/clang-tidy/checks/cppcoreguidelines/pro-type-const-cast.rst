.. title:: clang-tidy - cppcoreguidelines-pro-type-const-cast

cppcoreguidelines-pro-type-const-cast
=====================================

This check flags all uses of ``const_cast`` in C++ code.

Modifying a variable that was declared const is undefined behavior, even with
``const_cast``.

This rule is part of the `Type safety (Type 3)
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Pro-type-constcast>`_
profile from the C++ Core Guidelines.
