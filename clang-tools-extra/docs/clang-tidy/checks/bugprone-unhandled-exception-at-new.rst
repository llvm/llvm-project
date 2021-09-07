.. title:: clang-tidy - bugprone-unhandled-exception-at-new

bugprone-unhandled-exception-at-new
===================================

Finds calls to ``new`` with missing exception handler for ``std::bad_alloc``.

.. code-block:: c++

  int *f() noexcept {
    int *p = new int[1000];
    // ...
    return p;
  }

Calls to ``new`` can throw exceptions of type ``std::bad_alloc`` that should
be handled by the code. Alternatively, the nonthrowing form of ``new`` can be
used. The check verifies that the exception is handled in the function
that calls ``new``, unless a nonthrowing version is used or the exception
is allowed to propagate out of the function (exception handler is checked for
types ``std::bad_alloc``, ``std::exception``, and catch-all handler).
The check assumes that any user-defined ``operator new`` is either
``noexcept`` or may throw an exception of type ``std::bad_alloc`` (or derived
from it). Other exception types or exceptions occurring in the objects's
constructor are not taken into account.
