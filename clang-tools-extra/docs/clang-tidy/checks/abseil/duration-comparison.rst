.. title:: clang-tidy - abseil-duration-comparison

abseil-duration-comparison
==========================

Checks for comparisons which should be in the ``absl::Duration`` domain instead
of the floating point or integer domains.

N.B.: In cases where a ``Duration`` was being converted to an integer and then
compared against a floating-point value, truncation during the ``Duration``
conversion might yield a different result. In practice this is very rare, and
still indicates a bug which should be fixed.

Examples:

.. code-block:: c++

  // Original - Comparison in the floating point domain
  double x;
  absl::Duration d;
  if (x < absl::ToDoubleSeconds(d)) ...

  // Suggested - Compare in the absl::Duration domain instead
  if (absl::Seconds(x) < d) ...


  // Original - Comparison in the integer domain
  int x;
  absl::Duration d;
  if (x < absl::ToInt64Microseconds(d)) ...

  // Suggested - Compare in the absl::Duration domain instead
  if (absl::Microseconds(x) < d) ...
