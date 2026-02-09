.. title:: clang-tidy - bugprone-unsafe-to-allow-exceptions

bugprone-unsafe-to-allow-exceptions
===================================

Finds functions where throwing exceptions is unsafe but the function is still
marked as potentially throwing. Throwing exceptions from the following
functions can be problematic:

* Destructors
* Move constructors
* Move assignment operators
* The ``main()`` functions
* ``swap()`` functions
* ``iter_swap()`` functions
* ``iter_move()`` functions

A destructor throwing an exception may result in undefined behavior, resource
leaks or unexpected termination of the program. Throwing move constructor or
move assignment also may result in undefined behavior or resource leak. The
``swap()`` operations expected to be non throwing most of the cases and they
are always possible to implement in a non throwing way. Non throwing ``swap()``
operations are also used to create move operations. A throwing ``main()``
function also results in unexpected termination.

The check finds any of these functions if it is marked with ``noexcept(false)``
or ``throw(exception)``. This would indicate that the function is expected to
throw exceptions. Only the presence of these keywords is checked, not if the
function actually throws any exception. To check if the function actually
throws exception, the check :doc:`bugprone-exception-escape <exception-escape>`
can be used (but it does not warn if a function is explicitly marked as
throwing).

Options
-------

.. option:: CheckedSwapFunctions

   Semicolon-separated list of checked swap function names (where throwing
   exceptions is unsafe). These functions are checked if the parameter count is
   at least 1. Default value is `swap;iter_swap;iter_move`.
