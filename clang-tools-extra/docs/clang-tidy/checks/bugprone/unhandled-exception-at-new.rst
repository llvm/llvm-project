.. title:: clang-tidy - bugprone-unhandled-exception-at-new

bugprone-unhandled-exception-at-new
===================================

Finds calls to ``new`` with missing exception handler for ``std::bad_alloc``.

Calls to ``new`` may throw exceptions of type ``std::bad_alloc`` that should
be handled. Alternatively, the nonthrowing form of ``new`` can be
used. The check verifies that the exception is handled in the function
that calls ``new``.

If a nonthrowing version is used or the exception is allowed to propagate out
of the function no warning is generated.

The exception handler is checked if it catches a ``std::bad_alloc`` or
``std::exception`` exception type, or all exceptions (catch-all).
The check assumes that any user-defined ``operator new`` is either
``noexcept`` or may throw an exception of type ``std::bad_alloc`` (or one
derived from it). Other exception class types are not taken into account.

.. code-block:: c++

  int *f() noexcept {
    int *p = new int[1000]; // warning: missing exception handler for allocation failure at 'new'
    // ...
    return p;
  }

.. code-block:: c++

  int *f1() { // not 'noexcept'
    int *p = new int[1000]; // no warning: exception can be handled outside
                            // of this function
    // ...
    return p;
  }

  int *f2() noexcept {
    try {
      int *p = new int[1000]; // no warning: exception is handled
      // ...
      return p;
    } catch (std::bad_alloc &) {
      // ...
    }
    // ...
  }

  int *f3() noexcept {
    int *p = new (std::nothrow) int[1000]; // no warning: "nothrow" is used
    // ...
    return p;
  }

