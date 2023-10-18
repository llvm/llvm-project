.. title:: clang-tidy - cppcoreguidelines-pro-type-const-cast

cppcoreguidelines-pro-type-const-cast
=====================================

Imposes limitations on the use of ``const_cast`` within C++ code. It depends on
the :option:`StrictMode` option setting to determine whether it should flag all
instances of ``const_cast`` or only those that remove the ``const`` qualifier.

Modifying a variable that has been declared as ``const`` in C++ is generally
considered undefined behavior, and this remains true even when using
``const_cast``. In C++, the ``const`` qualifier indicates that a variable is
intended to be read-only, and the compiler enforces this by disallowing any
attempts to change the value of that variable.

This rule is part of the `Type safety (Type 3)
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Pro-type-constcast>`_
profile and `ES.50: Don’t cast away const
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#es50-dont-cast-away-const>`_
rule from the C++ Core Guidelines.

Options
-------

.. option:: StrictMode

  When this setting is set to `true`, it means that any usage of ``const_cast``
  is not allowed. On the other hand, when it's set to `false`, it permits
  casting to ``const`` types. By default, this setting is `false`.
