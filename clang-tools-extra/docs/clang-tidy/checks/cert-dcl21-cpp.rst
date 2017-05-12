.. title:: clang-tidy - cert-dcl21-cpp

cert-dcl21-cpp
==============

This check flags postfix ``operator++`` and ``operator--`` declarations
if the return type is not a const object. This also warns if the return type
is a reference type.

This check corresponds to the CERT C++ Coding Standard recommendation
`DCL21-CPP. Overloaded postfix increment and decrement operators should return a const object
<https://www.securecoding.cert.org/confluence/display/cplusplus/DCL21-CPP.+Overloaded+postfix+increment+and+decrement+operators+should+return+a+const+object>`_.
