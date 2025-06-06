.. title:: clang-tidy - performance-bool-bitwise-operation

performance-bool-bitwise-operation
==================================

Finds potentially inefficient use of bitwise operators such as ``&``,  ``|`` 
and their compound analogues on boolean values where logical operators like 
``&&`` and ``||`` would be more appropriate.

Bitwise operations on booleans can incur unnecessary performance overhead due 
to implicit integer conversions and missed short-circuit evaluation.

.. code-block:: c++

  bool invalid = false;
  invalid |= x > limit.x; // warning: use logical operator instead of bitwise one for bool
  invalid |= y > limit.y; // warning: use logical operator instead of bitwise one for bool
  invalid |= z > limit.z; // warning: use logical operator instead of bitwise one for bool
  if (invalid) {
    // error handling
  }

These 3 warnings suggest to assign result of logical ``||`` operation instead 
of using ``|=`` operator:

.. code-block:: c++

  bool invalid = false;
  invalid = invalid || x > limit.x;
  invalid = invalid || y > limit.x;
  invalid = invalid || z > limit.z;
  if (invalid) {
    // error handling
  }

Options
-------

.. option:: StrictMode

    Disabling this option promotes more fixit hints even when they might
    change evaluation order or skip side effects. Default value is `true`.