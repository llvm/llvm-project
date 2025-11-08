.. title:: clang-tidy - readability-constant-operand-order

readability-constant-operand-order
==================================

Warns when a constant appears on the non-preferred side of a supported binary
operator and offers a fix-it to swap operands (and invert the operator for
``<``, ``>``, ``<=``, ``>=``).

Examples
--------

.. code-block:: c++

  // Before
  if (nullptr == p) { /* ... */ }
  if (0 < x) { /* ... */ }

  // After
  if (p == nullptr) { /* ... */ }
  if (x > 0) { /* ... */ }

Options
-------

.. option:: PreferredConstantSide (string)

   Either ``Left`` or ``Right``. Default: ``Right``.

.. option:: BinaryOperators (string)

   Comma-separated list of operators to check. Default: ``==,!=,<,<=,>,>=``.
