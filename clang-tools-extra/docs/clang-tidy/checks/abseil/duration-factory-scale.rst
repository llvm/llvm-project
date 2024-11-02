.. title:: clang-tidy - abseil-duration-factory-scale

abseil-duration-factory-scale
=============================

Checks for cases where arguments to ``absl::Duration`` factory functions are
scaled internally and could be changed to a different factory function. This
check also looks for arguments with a zero value and suggests using
``absl::ZeroDuration()`` instead.

Examples:

.. code-block:: c++

  // Original - Internal multiplication.
  int x;
  absl::Duration d = absl::Seconds(60 * x);

  // Suggested - Use absl::Minutes instead.
  absl::Duration d = absl::Minutes(x);


  // Original - Internal division.
  int y;
  absl::Duration d = absl::Milliseconds(y / 1000.);

  // Suggested - Use absl:::Seconds instead.
  absl::Duration d = absl::Seconds(y);


  // Original - Zero-value argument.
  absl::Duration d = absl::Hours(0);

  // Suggested = Use absl::ZeroDuration instead
  absl::Duration d = absl::ZeroDuration();
