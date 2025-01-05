.. title:: clang-tidy - bugprone-undefined-sprintf-overlap

bugprone-undefined-sprintf-overlap
==================================

Warns if any arguments to the ``sprintf`` family of functions overlap with the
destination buffer (the first argument).

.. code-block:: c++

    char buf[20] = {"hi"};
    sprintf(buf, "%s%d", buf, 0);

C99 and POSIX.1-2001 states that if copying were to take place between objects
that overlap, the result is undefined.

Options
-------

.. option:: SprintfRegex

   A regex specifying the ``sprintf`` family of functions to match on. By default,
   this is `(::std)?::sn?printf`.
