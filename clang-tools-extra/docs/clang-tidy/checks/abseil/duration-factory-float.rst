.. title:: clang-tidy - abseil-duration-factory-float

abseil-duration-factory-float
=============================

Checks for cases where the floating-point overloads of various
``absl::Duration`` factory functions are called when the more-efficient
integer versions could be used instead.

This check will not suggest fixes for literals which contain fractional
floating point values or non-literals. It will suggest removing
superfluous casts.

Examples:

.. code-block:: c++

  // Original - Providing a floating-point literal.
  absl::Duration d = absl::Seconds(10.0);

  // Suggested - Use an integer instead.
  absl::Duration d = absl::Seconds(10);


  // Original - Explicitly casting to a floating-point type.
  absl::Duration d = absl::Seconds(static_cast<double>(10));

  // Suggested - Remove the explicit cast
  absl::Duration d = absl::Seconds(10);
