.. title:: clang-tidy - bugprone-macro-condition

bugprone-macro-condition
========================

Warns about inconsistent macro usage in preprocessor conditions.

Given the following code:

.. code-block:: c++

  #define USE_FOO 0
  // ...
  #if defined(USE_FOO)
    // ...
  #endif

Here `USE_FOO` is defined to a value that would evaluate to false in a
preprocessor condition, but checked for definition and not for its value.
Was the intention to evaluate `USE_FOO` for a `true` expression, or was
the intention to merely check whether or not the macro was defined?

If the code later contains:

.. code-block:: c++

  #if USE_FOO
    // ...
  #endif

Then the suspicions are raised further.

The following scenarios result in warnings; no fixes
are offered as the scenarios are all ambiguous.

- A macro is defined to a value,
  but it is checked for definition in a condition.
- A macro in a condition is checked for definition in one location
  and for value in another location.
