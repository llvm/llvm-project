.. title:: clang-tidy - abseil-duration-unnecessary-conversion

abseil-duration-unnecessary-conversion
======================================

Finds and fixes cases where ``absl::Duration`` values are being converted to
numeric types and back again.

Examples:

.. code-block:: c++

  // Original - Conversion to double and back again
  absl::Duration d1;
  absl::Duration d2 = absl::Seconds(absl::ToDoubleSeconds(d1));

  // Suggestion - Remove unnecessary conversions
  absl::Duration d2 = d1;


  // Original - Conversion to integer and back again
  absl::Duration d1;
  absl::Duration d2 = absl::Hours(absl::ToInt64Hours(d1));

  // Suggestion - Remove unnecessary conversions
  absl::Duration d2 = d1;

Note: Converting to an integer and back to an ``absl::Duration`` might be a
truncating operation if the value is not aligned to the scale of conversion.
In the rare case where this is the intended result, callers should use
``absl::Trunc`` to truncate explicitly.
