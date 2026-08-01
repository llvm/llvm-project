.. title:: clang-tidy - readability-use-span-first-last

readability-use-span-first-last
===============================

Suggests using ``std::span::first()`` and ``std::span::last()`` member
functions instead of equivalent ``subspan()`` calls. These dedicated methods
were added to C++20 to provide more expressive alternatives to common subspan
operations.

Covered scenarios:

=========================== ==============
Expression                  Replacement
--------------------------- --------------
``s.subspan(0, n)``         ``s.first(n)``
``s.subspan(s.size() - n)`` ``s.last(n)``
=========================== ==============

Non-zero offset with count (like ``subspan(1, n)``) or offset-only calls
(like ``subspan(n)``) have no clearer equivalent using ``first()`` or
``last()``, so these cases are not transformed.

This check is enabled for C++20 or later.
