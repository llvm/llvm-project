.. title:: clang-tidy - misc-misplaced-widening-cast

misc-misplaced-widening-cast
============================

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

Options
-------

.. option:: CheckImplicitCasts

   If non-zero, enables detection of implicit casts. Default is non-zero.
