.. title:: clang-tidy - abseil-time-subtraction

abseil-time-subtraction
=======================

Finds and fixes ``absl::Time`` subtraction expressions to do subtraction
in the Time domain instead of the numeric domain.

There are two cases of Time subtraction in which deduce additional type
information:
 - When the result is an ``absl::Duration`` and the first argument is an
   ``absl::Time``.
 - When the second argument is a ``absl::Time``.

In the first case, we must know the result of the operation, since without that
the second operand could be either an ``absl::Time`` or an ``absl::Duration``.
In the second case, the first operand *must* be an ``absl::Time``, because
subtracting an ``absl::Time`` from an ``absl::Duration`` is not defined.

Examples:

.. code-block:: c++
  int x;
  absl::Time t;

  // Original - absl::Duration result and first operand is a absl::Time.
  absl::Duration d = absl::Seconds(absl::ToUnixSeconds(t) - x);

  // Suggestion - Perform subtraction in the Time domain instead.
  absl::Duration d = t - absl::FromUnixSeconds(x);


  // Original - Second operand is an absl::Time.
  int i = x - absl::ToUnixSeconds(t);

  // Suggestion - Perform subtraction in the Time domain instead.
  int i = absl::ToInt64Seconds(absl::FromUnixSeconds(x) - t);
