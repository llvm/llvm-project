.. title:: clang-tidy - readability-redundant-casting

readability-redundant-casting
=============================

Detects explicit type casting operations that involve the same source and
destination types, and subsequently recommend their removal. Covers a range of
explicit casting operations, including ``static_cast``, ``const_cast``, C-style
casts, and ``reinterpret_cast``. Its primary objective is to enhance code
readability and maintainability by eliminating unnecessary type casting.

.. code-block:: c++

  int value = 42;
  int result = static_cast<int>(value);

In this example, the ``static_cast<int>(value)`` is redundant, as it performs
a cast from an ``int`` to another ``int``.

Casting operations involving constructor conversions, user-defined conversions,
functional casts, type-dependent casts, casts between distinct type aliases that
refer to the same underlying type, as well as bitfield-related casts and casts
directly from lvalue to rvalue, are all disregarded by the check.

Options
-------

.. option:: IgnoreMacros

   If set to `true`, the check will not give warnings inside macros. Default
   is `true`.

.. option:: IgnoreTypeAliases

   When set to `false`, the check will consider type aliases, and when set to
   `true`, it will resolve all type aliases and operate on the underlying
   types. Default is `false`.
