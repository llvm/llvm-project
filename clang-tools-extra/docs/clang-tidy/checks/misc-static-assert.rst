.. title:: clang-tidy - misc-static-assert

misc-static-assert
==================

"cert-dcl03-c" redirects here as an alias for this checker.

Replaces ``assert()`` with ``static_assert()`` if the condition is evaluatable
at compile time.

The condition of ``static_assert()`` is evaluated at compile time which is
safer and more efficient.
