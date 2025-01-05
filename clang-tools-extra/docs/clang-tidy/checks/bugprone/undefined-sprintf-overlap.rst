.. title:: clang-tidy - bugprone-undefined-sprintf-overlap

bugprone-undefined-sprintf-overlap
==================================

Warns if any arguments to the ``sprintf`` family of functions overlap with the
destination buffer (the first argument).

.. code-block:: c++

    char buf[20] = {"hi"};
    sprintf(buf, "%s%d", buf, 0);

If copying takes place between objects that overlap, the behavior is undefined.
This is stated in the `C23/N3220 standard
<https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3096.pdf>`_
(sections 7.23.6.5 and 7.23.6.6), as well as the `POSIX.1-2024 standard
<https://pubs.opengroup.org/onlinepubs/9799919799/>`_.

Options
-------

.. option:: SprintfRegex

   A regex specifying the ``sprintf`` family of functions to match on. By default,
   this is `(::std)?::sn?printf`.
