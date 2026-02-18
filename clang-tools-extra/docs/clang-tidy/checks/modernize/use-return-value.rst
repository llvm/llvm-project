.. title:: clang-tidy - modernize-use-return-value

modernize-use-return-value
==========================

Finds functions that return ``void`` and have a single non-const
reference output parameter, suggesting that they return the value
directly instead.

Returning values instead of using output parameters is generally
preferred in modern C++. This is clearer, works better with ``auto``
and structured bindings, and enables copy/move elision.

.. code-block:: c++

  // Before
  void getResult(int &Out) {
    Out = compute();
  }

  // After
  int getResult() {
    return compute();
  }

The check will not flag a function if it:

- returns non-void,
- has zero or more than one non-const reference parameter,
- never assigns to the output parameter,
- has an output parameter of abstract or array type,
- is a virtual method, or
- has an unnamed output parameter.

.. note::

   This check does not provide fix-its because changing the
   return type requires updating all call sites, which may
   span multiple translation units.
