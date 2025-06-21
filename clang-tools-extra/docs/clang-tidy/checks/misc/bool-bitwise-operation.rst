.. title:: clang-tidy - misc-bool-bitwise-operation

misc-bool-bitwise-operation
===========================

Finds potentially inefficient use of bitwise operators such as ``&``,  ``|`` 
and their compound analogues on Boolean values where logical operators like 
``&&`` and ``||`` would be more appropriate.

Bitwise operations on Booleans can incur unnecessary performance overhead due 
to implicit integer conversions and missed short-circuit evaluation.

.. code-block:: c++

  bool invalid = false;
  invalid |= x > limit.x; // warning: use logical operator '||' for boolean variable 'invalid' instead of bitwise operator '|='
  invalid |= y > limit.y; // warning: use logical operator '||' for boolean variable 'invalid' instead of bitwise operator '|='
  invalid |= z > limit.z; // warning: use logical operator '||' for boolean variable 'invalid' instead of bitwise operator '|='
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

    Disabling this option promotes more fix-it hints even when they might
    change evaluation order or skip side effects. Default value is `true`.

.. option:: IgnoreMacros

    Enabling this option hides the warning message in a situation where
    it is not possible to change a bitwise operator to a logical one due
    to a macro in the expression body. Default value is `false`.
