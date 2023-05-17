.. title:: clang-tidy - misc-new-delete-overloads

misc-new-delete-overloads
=========================

`cert-dcl54-cpp` redirects here as an alias for this check.

The check flags overloaded operator ``new()`` and operator ``delete()``
functions that do not have a corresponding free store function defined within
the same scope.
For instance, the check will flag a class implementation of a non-placement
operator ``new()`` when the class does not also define a non-placement operator
``delete()`` function as well.

The check does not flag implicitly-defined operators, deleted or private
operators, or placement operators.

This check corresponds to CERT C++ Coding Standard rule `DCL54-CPP. Overload allocation and deallocation functions as a pair in the same scope
<https://www.securecoding.cert.org/confluence/display/cplusplus/DCL54-CPP.+Overload+allocation+and+deallocation+functions+as+a+pair+in+the+same+scope>`_.
