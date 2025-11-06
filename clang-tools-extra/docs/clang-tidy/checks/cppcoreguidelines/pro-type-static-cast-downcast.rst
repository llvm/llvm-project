.. title:: clang-tidy - cppcoreguidelines-pro-type-static-cast-downcast

cppcoreguidelines-pro-type-static-cast-downcast
===============================================

This check flags all usages of ``static_cast``, where a base class is casted to
a derived class. In those cases, a fix-it is provided to convert the cast to a
``dynamic_cast``.

Use of these casts can violate type safety and cause the program to access a
variable that is actually of type ``X`` to be accessed as if it were of an
unrelated type ``Z``.

This rule is part of the `Type safety (Type.2)
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Pro-type-downcast>`_
profile from the C++ Core Guidelines.

Options
-------

.. option:: StrictMode

  When set to `false`, no warnings are emitted for casts on non-polymorphic
  types. Default is `true`.
