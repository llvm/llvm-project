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
  invalid |= x > limit.x; // warning: use logical operator '||' for boolean semantics instead of bitwise operator '|='
                          //   400 |     invalid |= x > limit.x;
                          //       |             ^~
                          //       |             = invalid ||
  invalid |= y > limit.y; // warning: use logical operator '||' for boolean semantics instead of bitwise operator '|='
                          //   401 |     invalid |= y > limit.y;
                          //       |             ^~
                          //       |             = invalid ||
  invalid |= z > limit.z; // warning: use logical operator '||' for boolean semantics instead of bitwise operator '|='
                          //   402 |     invalid |= z > limit.z;
                          //       |             ^~
                          //       |             = invalid ||
  if (invalid) {
    // error handling
  }

These 3 warnings suggest assigning the result of a logical ``||`` operation 
instead of using the ``|=`` operator:

.. code-block:: c++

  bool invalid = false;
  invalid = invalid || x > limit.x;
  invalid = invalid || y > limit.x;
  invalid = invalid || z > limit.z;
  if (invalid) {
    // error handling
  }

It is not always a safe transformation though. The following case will warn
without fix-it to preserve the semantics.

.. code-block:: c++

  volatile bool invalid = false;
  invalid |= x > limit.x; // warning: use logical operator '||' for boolean semantics instead of bitwise operator '|='

Limitations
-----------

* Bitwise operators inside templates aren't guaranteed to match.

.. code-block:: c++

     template <typename X>
     void f(X a, X b) {
         a | b; // the warning may not be emitted
     }

Options
-------

.. option:: UnsafeMode

    Provide more fix-it hints even when they might change evaluation order or
    skip side effects. Default value is `false`.

.. option:: IgnoreMacros

    Don't warn if a macro inside the expression body prevents replacing a
    bitwise operator with a logical one. Default value is `false`.