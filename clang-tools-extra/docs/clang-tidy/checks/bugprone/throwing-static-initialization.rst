.. title:: clang-tidy - bugprone-throwing-static-initialization

bugprone-throwing-static-initialization
=======================================

Finds all ``static`` or ``thread_local`` variable declarations where the
initializer for the object may throw an exception.

References
----------

This check corresponds to the CERT C++ Coding Standard rule
`ERR58-CPP. Handle all exceptions thrown before main() begins executing
<https://www.securecoding.cert.org/confluence/display/cplusplus/ERR58-CPP.+Handle+all+exceptions+thrown+before+main%28%29+begins+executing>`_.
