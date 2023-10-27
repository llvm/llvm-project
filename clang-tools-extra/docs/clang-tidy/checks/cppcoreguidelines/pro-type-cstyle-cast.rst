.. title:: clang-tidy - cppcoreguidelines-pro-type-cstyle-cast

cppcoreguidelines-pro-type-cstyle-cast
======================================

This check flags all use of C-style casts that perform a ``static_cast``
downcast, ``const_cast``, or ``reinterpret_cast``.

Use of these casts can violate type safety and cause the program to access a
variable that is actually of type X to be accessed as if it were of an unrelated
type Z. Note that a C-style ``(T)expression`` cast means to perform the first of
the following that is possible: a ``const_cast``, a ``static_cast``, a
``static_cast`` followed by a ``const_cast``, a ``reinterpret_cast``, or a
``reinterpret_cast`` followed by a ``const_cast``. This rule bans
``(T)expression`` only when used to perform an unsafe cast.

This rule is part of the `Type safety (Type.4)
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Pro-type-cstylecast>`_
profile from the C++ Core Guidelines.
