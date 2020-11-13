.. title:: clang-tidy - bugprone-signal-handler

bugprone-signal-handler
=======================

Finds functions registered as signal handlers that call non asynchronous-safe
functions. Any function that cannot be determined to be an asynchronous-safe
function call is assumed to be non-asynchronous-safe by the checker,
including user functions for which only the declaration is visible.
User function calls with visible definition are checked recursively.
The check handles only C code.

The minimal list of asynchronous-safe system functions is:
``abort()``, ``_Exit()``, ``quick_exit()`` and ``signal()``
(for ``signal`` there are additional conditions that are not checked).
The check accepts only these calls as asynchronous-safe.

This check corresponds to the CERT C Coding Standard rule
`SIG30-C. Call only asynchronous-safe functions within signal handlers
<https://www.securecoding.cert.org/confluence/display/c/SIG30-C.+Call+only+asynchronous-safe+functions+within+signal+handlers>`_.
