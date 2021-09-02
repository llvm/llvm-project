.. title:: clang-tidy - bugprone-signal-handler

bugprone-signal-handler
=======================

Finds functions registered as signal handlers that call non asynchronous-safe
functions. Any function that cannot be determined to be an asynchronous-safe
function call is assumed to be non-asynchronous-safe by the checker,
including user functions for which only the declaration is visible.
User function calls with visible definition are checked recursively.
The check handles only C code. Only the function names are considered and the
fact that the function is a system-call, but no other restrictions on the
arguments passed to the functions (the ``signal`` call is allowed without
restrictions).

This check corresponds to the CERT C Coding Standard rule
`SIG30-C. Call only asynchronous-safe functions within signal handlers
<https://www.securecoding.cert.org/confluence/display/c/SIG30-C.+Call+only+asynchronous-safe+functions+within+signal+handlers>`_
and has an alias name ``cert-sig30-c``.

.. option:: AsyncSafeFunctionSet

  Selects which set of functions is considered as asynchronous-safe
  (and therefore allowed in signal handlers). Value ``minimal`` selects
  a minimal set that is defined in the CERT SIG30-C rule and includes functions
  ``abort()``, ``_Exit()``, ``quick_exit()`` and ``signal()``. Value ``POSIX``
  selects a larger set of functions that is listed in POSIX.1-2017 (see `this
  link
  <https://pubs.opengroup.org/onlinepubs/9699919799/functions/V2_chap02.html#tag_15_04_03>`_
  for more information).
  The function ``quick_exit`` is not included in the shown list. It is
  assumable that  the reason is that the list was not updated for C11.
  The checker includes ``quick_exit`` in the set of safe functions.
  Functions registered as exit handlers are not checked.
  
  Default is ``POSIX``.

