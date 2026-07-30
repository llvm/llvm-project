.. title:: clang-tidy - bugprone-cerrno

bugprone-cerrno
===============

Warns if you declare an ``extern int`` variable named ``errno`` instead of including ``<cerrno>``.
It is able to fix the problem by removing the line of the declaration and inserting ``#include <cerrno>``
at the top of the file.

For further reading, see `the page of SEI CERT C Coding Standard
<https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/rules/miscellaneous-msc/msc38-c/>`_.

Example:

.. code-block:: c++

    extern int errno;
