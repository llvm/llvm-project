.. title:: clang-tidy - abseil-duration-division

abseil-duration-division
========================

``absl::Duration`` arithmetic works like it does with integers. That means that
division of two ``absl::Duration`` objects returns an ``int64`` with any fractional
component truncated toward 0. See `this link <https://github.com/abseil/abseil-cpp/blob/29ff6d4860070bf8fcbd39c8805d0c32d56628a3/absl/time/time.h#L137>`_ for more information on arithmetic with ``absl::Duration``.

For example:

.. code-block:: c++

 absl::Duration d = absl::Seconds(3.5);
 int64 sec1 = d / absl::Seconds(1);     // Truncates toward 0.
 int64 sec2 = absl::ToInt64Seconds(d);  // Equivalent to division.
 assert(sec1 == 3 && sec2 == 3);

 double dsec = d / absl::Seconds(1);  // WRONG: Still truncates toward 0.
 assert(dsec == 3.0);

If you want floating-point division, you should use either the
``absl::FDivDuration()`` function, or one of the unit conversion functions such
as ``absl::ToDoubleSeconds()``. For example:

.. code-block:: c++

 absl::Duration d = absl::Seconds(3.5);
 double dsec1 = absl::FDivDuration(d, absl::Seconds(1));  // GOOD: No truncation.
 double dsec2 = absl::ToDoubleSeconds(d);                 // GOOD: No truncation.
 assert(dsec1 == 3.5 && dsec2 == 3.5);


This check looks for uses of ``absl::Duration`` division that is done in a
floating-point context, and recommends the use of a function that returns a
floating-point value.
