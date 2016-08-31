.. title:: clang-tidy - misc-suspicious-string-compare

misc-suspicious-string-compare
==============================

Find suspicious usage of runtime string comparison functions.
This check is valid in C and C++.

Checks for calls with implicit comparator and proposed to explicitly add it.

.. code-block:: c++

    if (strcmp(...))       // Implicitly compare to zero
    if (!strcmp(...))      // Won't warn
    if (strcmp(...) != 0)  // Won't warn

Checks that compare function results (i,e, ``strcmp``) are compared to valid
constant. The resulting value is

.. code::

    <  0    when lower than,
    >  0    when greater than,
    == 0    when equals.

A common mistake is to compare the result to `1` or `-1`.

.. code-block:: c++

    if (strcmp(...) == -1)  // Incorrect usage of the returned value.

Additionally, the check warns if the results value is implicitly cast to a
*suspicious* non-integer type. It's happening when the returned value is used in
a wrong context.

.. code-block:: c++

    if (strcmp(...) < 0.)  // Incorrect usage of the returned value.
