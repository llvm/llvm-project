.. title:: clang-tidy - abseil-duration-addition

abseil-duration-addition
========================

Check for cases where addition should be performed in the ``absl::Time`` domain.
When adding two values, and one is known to be an ``absl::Time``, we can infer
that the other should be interpreted as an ``absl::Duration`` of a similar
scale, and make that inference explicit.

Examples:

.. code-block:: c++

  // Original - Addition in the integer domain
  int x;
  absl::Time t;
  int result = absl::ToUnixSeconds(t) + x;

  // Suggestion - Addition in the absl::Time domain
  int result = absl::TounixSeconds(t + absl::Seconds(x));
