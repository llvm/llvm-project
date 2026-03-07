.. title:: clang-tidy - misc-bool-bitwise-operation

misc-bool-bitwise-operation
===========================

Finds potentially inefficient use of bitwise operators such as ``&``,  ``|``
and their compound analogues on Boolean values where logical operators like
``&&`` and ``||`` would be more appropriate.

Bitwise operations on Booleans can incur unnecessary performance overhead due
to implicit integer conversions and missed short-circuit evaluation. They also
contradict the principle of least astonishment, as logical operators are the
expected and idiomatic way to work with Boolean values.

.. code-block:: c++

  // Before:
  bool invalid = false;
  invalid |= x > limit.x;
  invalid |= y > limit.y;
  invalid |= z > limit.z;
  if (invalid) {
    // error handling
  }

  // After:
  bool invalid = false;
  invalid = invalid || (x > limit.x);
  invalid = invalid || (y > limit.y);
  invalid = invalid || (z > limit.z);
  if (invalid) {
    // error handling
  }

It is not always a safe transformation though. The following case will warn
without fix-it to preserve the semantics.

.. code-block:: c++

  volatile bool invalid = false;
  invalid |= x > limit.x;

Limitations
-----------

* Bitwise operators inside templates aren't guaranteed to match.

.. code-block:: c++

     template <typename X>
     void f(X a, X b) {
         a | b; // the warning may not be emitted
     }

* For compound operators (``&=``, ``|=``), the left-hand side (LHS) must be
  simple. Only the following are supported:

  * Variable references (``declRefExpr``)
  * Member access expressions (``memberExpr``)
  * Builtin dereferencing (``*``) of the above

  This limitation exists because the fix-it needs to duplicate the LHS on the
  right-hand side of the transformed assignment. Complex expressions cannot be
  safely duplicated.

.. code-block:: c++

     bool a, b;
     struct S { bool flag; } s;
     bool *p = &a;

     a |= b;        // Supported: simple variable reference
     s.flag |= b;   // Supported: member access
     *p |= b;       // Supported: builtin dereferencing

     (a ? b : a) |= b;  // Not supported: complex expression

Options
-------

.. option:: UnsafeMode

    Provide more fix-it hints even when they might change evaluation order or
    skip side effects. Default value is `false`.

.. option:: IgnoreMacros

    Don't warn if a macro inside the expression body prevents replacing a
    bitwise operator with a logical one. Default value is `false`.

.. option:: StrictMode

    When enabled, show warnings even when fix-it hints cannot be generated
    (e.g., for volatile operands or expressions with side effects). When
    disabled, only show warnings when fix-it hints are available. Default
    value is `true`.

.. option:: ParenCompounds

    When enabled, add parentheses around the right-hand side (RHS) of compound
    operators (``&=``, ``|=``) when transforming them to logical operators,
    except when the RHS already uses the same logical operator or is already
    parenthesized. This helps improve readability, avoid potential
    misunderstandings of precedence, and prevent `-WParens` warnings from
    compilers. Default value is `true`.

.. code-block:: c++

  bool a, b, c;
  a &= b | c;  // With ParenCompounds=true:  a = a && (b || c);
                // With ParenCompounds=false: a = a && b || c;

  a &= b && c; // Always: a = a && b && c; (no parentheses needed)

  a &= b || c; // Always: a = a && (b || c); (parentheses for precedence)
