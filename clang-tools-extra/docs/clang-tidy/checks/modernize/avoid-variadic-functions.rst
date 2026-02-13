.. title:: clang-tidy - modernize-avoid-variadic-functions

modernize-avoid-variadic-functions
==================================

Find all function definitions (but not declarations) of C-style variadic
functions.

Instead of C-style variadic functions, C++ function parameter pack should be
used.


References
----------

This check corresponds to the CERT C++ Coding Standard rule
`DCL50-CPP. Do not define a C-style variadic function
<https://www.securecoding.cert.org/confluence/display/cplusplus/DCL50-CPP.+Do+not+define+a+C-style+variadic+function>`_.
