.. title:: clang-tidy - bugprone-misplaced-widening-cast

bugprone-misplaced-widening-cast
================================

This check will warn when there is a cast of a calculation result to a bigger
type. If the intention of the cast is to avoid loss of precision then the cast
is misplaced, and there can be loss of precision. Otherwise the cast is
ineffective.

Example code:

.. code-block:: c++

    long f(int x) {
        return (long)(x * 1000);
    }

The result ``x * 1000`` is first calculated using ``int`` precision. If the
result exceeds ``int`` precision there is loss of precision. Then the result is
casted to ``long``.

If there is no loss of precision then the cast can be removed or you can
explicitly cast to ``int`` instead.

If you want to avoid loss of precision then put the cast in a proper location,
for instance:

.. code-block:: c++

    long f(int x) {
        return (long)x * 1000;
    }

Implicit casts
--------------

Forgetting to place the cast at all is at least as dangerous and at least as
common as misplacing it. If :option:`CheckImplicitCasts` is enabled the check
also detects these cases, for instance:

.. code-block:: c++

    long f(int x) {
        return x * 1000;
    }

Floating point
--------------

Currently warnings are only written for integer conversion. No warning is
written for this code:

.. code-block:: c++

    double f(float x) {
        return (double)(x * 10.0f);
    }

Constexpr operands
------------------

If :option:`IgnoreConstexprOverflowProven` is set to `true` and the calculation
operand is a ``constexpr`` variable, the check evaluates its compile-time value.
If the value provably does not overflow the narrower type and is non-negative,
no warning is issued, since the cast is then not actually masking any loss of
precision:

.. code-block:: c++

    void bar(std::size_t);

    void f() {
        constexpr int x = 256;
        bar(static_cast<std::size_t>(x * 2)); // No warning: x * 2 == 512,
                                               // which safely fits in int.
    }

A plain ``const`` (not ``constexpr``) variable does not get this exception,
since the check does not evaluate non-``constexpr`` values as compile-time
constants.

A ``constexpr`` value that results in a negative value or signed integer
overflow will still be diagnosed:

.. code-block:: c++

    void bar(std::size_t);

    void f() {
        constexpr int x = 2147483647;         // INT_MAX
        bar(static_cast<std::size_t>(x * 2)); // Still warns: calculation overflows int.
    }

Options
-------

.. option:: CheckImplicitCasts

   If `true`, enables detection of implicit casts. Default is `false`.

.. option:: IgnoreConstexprOverflowProven

   When set to `true`, the check evaluates the compile-time value of ``constexpr``
   calculation operands. If the value provably does not overflow the narrower
   type and is non-negative, the warning is suppressed because the widening cast
   is not masking an unintended truncation. Default is `false`.
