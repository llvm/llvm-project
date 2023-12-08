.. title:: clang-tidy - abseil-duration-subtraction

abseil-duration-subtraction
===========================

Checks for cases where subtraction should be performed in the
``absl::Duration`` domain. When subtracting two values, and the first one is
known to be a conversion from ``absl::Duration``, we can infer that the second
should also be interpreted as an ``absl::Duration``, and make that inference
explicit.

Examples:

.. code-block:: c++

  // Original - Subtraction in the double domain
  double x;
  absl::Duration d;
  double result = absl::ToDoubleSeconds(d) - x;

  // Suggestion - Subtraction in the absl::Duration domain instead
  double result = absl::ToDoubleSeconds(d - absl::Seconds(x));

  // Original - Subtraction of two Durations in the double domain
  absl::Duration d1, d2;
  double result = absl::ToDoubleSeconds(d1) - absl::ToDoubleSeconds(d2);

  // Suggestion - Subtraction in the absl::Duration domain instead
  double result = absl::ToDoubleSeconds(d1 - d2);


Note: As with other ``clang-tidy`` checks, it is possible that multiple fixes
may overlap (as in the case of nested expressions), so not all occurrences can
be transformed in one run. In particular, this may occur for nested subtraction
expressions. Running ``clang-tidy`` multiple times will find and fix these
overlaps.
