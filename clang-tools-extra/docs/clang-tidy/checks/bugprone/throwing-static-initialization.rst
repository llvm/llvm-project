.. title:: clang-tidy - bugprone-throwing-static-initialization

bugprone-throwing-static-initialization
=======================================

Finds all ``static`` or ``thread_local`` variable declarations where the
initializer for the object may throw an exception.

Options
-------

.. option:: IgnoredTypes

This option makes it possible to ignore specific types used at variable
declarations. It may contain a semicolon-separated list of regular expressions.
Declarations with a type that is matched by this list are excluded from
producing warnings by the check. The entries of the list are matched as
substrings of the type name.

This option contains by default an empty string.

References
----------

This check corresponds to the CERT C++ Coding Standard rule
`ERR58-CPP. Handle all exceptions thrown before main() begins executing
<https://www.securecoding.cert.org/confluence/display/cplusplus/ERR58-CPP.+Handle+all+exceptions+thrown+before+main%28%29+begins+executing>`_.
