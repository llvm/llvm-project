.. title:: clang-tidy - abseil-time-comparison

abseil-time-comparison
======================

Prefer comparisons in the ``absl::Time`` domain instead of the integer domain.

N.B.: In cases where an ``absl::Time`` is being converted to an integer,
alignment may occur. If the comparison depends on this alingment, doing the
comparison in the ``absl::Time`` domain may yield a different result. In
practice this is very rare, and still indicates a bug which should be fixed.

Examples:

.. code-block:: c++

  // Original - Comparison in the integer domain
  int x;
  absl::Time t;
  if (x < absl::ToUnixSeconds(t)) ...

  // Suggested - Compare in the absl::Time domain instead
  if (absl::FromUnixSeconds(x) < t) ...
