.. title:: clang-tidy - abseil-duration-conversion-cast

abseil-duration-conversion-cast
===============================

Checks for casts of ``absl::Duration`` conversion functions, and recommends
the right conversion function instead.

Examples:

.. code-block:: c++

  // Original - Cast from a double to an integer
  absl::Duration d;
  int i = static_cast<int>(absl::ToDoubleSeconds(d));

  // Suggested - Use the integer conversion function directly.
  int i = absl::ToInt64Seconds(d);


  // Original - Cast from a double to an integer
  absl::Duration d;
  double x = static_cast<double>(absl::ToInt64Seconds(d));

  // Suggested - Use the integer conversion function directly.
  double x = absl::ToDoubleSeconds(d);


Note: In the second example, the suggested fix could yield a different result,
as the conversion to integer could truncate.  In practice, this is very rare,
and you should use ``absl::Trunc`` to perform this operation explicitly instead.
