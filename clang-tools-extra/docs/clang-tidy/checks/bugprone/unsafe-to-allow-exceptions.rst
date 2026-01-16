.. title:: clang-tidy - bugprone-unsafe-to-allow-exceptions

bugprone-unsafe-to-allow-exceptions
===================================

Finds functions where throwing exceptions is unsafe but the function is still
marked as throwable. Throwing exceptions from the following functions can be
problematic:

* Destructors
* Move constructors
* Move assignment operators
* The ``main()`` functions
* ``swap()`` functions
* ``iter_swap()`` functions
* ``iter_move()`` functions

The check finds any of these functions if it is marked with ``noexcept(false)``
or ``throw(exception)``. This would indicate that the function is expected to
throw exceptions. Only the presence of these keywords is checked, not if the
function actually throws any exception.

Options
-------

.. option:: CheckedSwapFunctions

   Comma-separated list of checked swap function names (where throwing
   exceptions is unsafe).
   Default value is `swap,iter_swap,iter_move`.

