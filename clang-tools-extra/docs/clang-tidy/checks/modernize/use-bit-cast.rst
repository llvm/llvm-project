.. title:: clang-tidy - modernize-use-bit-cast

modernize-use-bit-cast
======================

Finds conservative object-to-object ``memcpy`` type punning that can be
rewritten as ``std::bit_cast`` in C++20 and later.

The check targets the common pattern of copying the full object representation
of one trivially copyable object into another trivially copyable object of a
different type:

.. code-block:: c++

  float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, &src, sizeof(src));

This is rewritten to:

.. code-block:: c++

  float src = 1.0f;
  unsigned int dst;
  dst = std::bit_cast<unsigned int>(src);

The fix intentionally replaces only the ``memcpy`` call. It does not fold a
preceding declaration into ``auto dst = ...`` because doing so can change the
construction behavior of the destination object.

It only matches direct named source and destination objects, or direct
field subobjects accessed through ``.``, ``->``, ``.*``, or ``->*``,
and only when:

* both object types are trivially copyable,
* neither object type is a pointer, array, function type, or
  volatile-qualified,
* the source and destination types differ,
* the copy size is expressed as ``sizeof(...)`` for either copied type, and
* the ``memcpy`` call appears in a discarded-value context, such as a statement
  body, the operand of an explicit ``(void)`` cast, or a comma subexpression
  whose value is discarded.

The check intentionally does not diagnose:

* pointer punning,
* array or buffer manipulation,
* macro expansions,
* dependent template cases,
* unevaluated contexts such as ``sizeof(memcpy(...))``,
* larger expressions where the ``memcpy`` value affects the enclosing
  expression, such as conditions or operands of unrelated operators,
* calls where the return value of ``memcpy`` is used, or
* unrelated overloads such as a user-defined ``memcpy``.

If needed, the fix also inserts ``#include <bit>``.

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, ``llvm`` or ``google``.
   Default is ``llvm``.
