.. title:: clang-tidy - bugprone-loop-variable-copied-then-modified

bugprone-loop-variable-copied-then-modified
===========================================

Detects when a loop variable is copied and then subsequently (possibly) modified
and suggests replacing with a reference or an explicit copy.

This pattern is considered bugprone because, frequently, programmers do not
realize that they are modifying a *copy* rather than an underlying value,
resulting in subtly erroneous code.

For instance, the following code attempts to null out a value in a map, but only
succeeds in nulling out a value in a *copy* of the map:

.. code-block:: c++

  for (auto target : target_map) {
    target.value = nullptr;
  }

The programmer is likely to have intended this code instead:

.. code-block:: c++

  for (auto& target : target_map) {
    target.value = nullptr;
  }

This code can be fixed in one of two ways:
  - In cases where the programmer did not intend to create a copy, they can
    convert the loop variable to a reference or a ``const`` reference. A
    fix-note message will provide a naive suggestion of how to achieve this,
    which works in most cases.
  - In cases where the intent is in fact to modify a copy, they may perform the
    copy explicitly, inside the body of the loop, and perform whatever
    operations they like on that copy.

This is a conservative check: in cases where it cannot be determined at compile
time whether or not a particular function modifies the variable, it assumes a
modification has ocurred and warns accordingly. However, in such cases, the
warning can still be suppressed by doing one of the actions described above.

Options
-------

.. option:: IgnoreInexpensiveVariables

   When `true`, this check will only alert on types that are expensive to copy.
   This will lead to fewer false positives, but will also overlook some
   instances where there may be an actual bug. Default is `false`.

.. option:: WarnOnlyOnAutoCopies

  When `true`, this check will only alert on `auto` types. Default is `false`.