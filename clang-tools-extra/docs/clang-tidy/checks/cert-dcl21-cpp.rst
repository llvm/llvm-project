.. title:: clang-tidy - cert-dcl21-cpp

cert-dcl21-cpp
==============

This check flags postfix ``operator++`` and ``operator--`` declarations
if the return type is not a const object. This also warns if the return type
is a reference type.

The object returned by a postfix increment or decrement operator is supposed
to be a snapshot of the object's value prior to modification. With such an
implementation, any modifications made to the resulting object from calling
operator++(int) would be modifying a temporary object. Thus, such an
implementation of a postfix increment or decrement operator should instead
return a const object, prohibiting accidental mutation of a temporary object.
Similarly, it is unexpected for the postfix operator to return a reference to
its previous state, and any subsequent modifications would be operating on a
stale object.

This check corresponds to the CERT C++ Coding Standard recommendation
DCL21-CPP. Overloaded postfix increment and decrement operators should return a
const object. However, all of the CERT recommendations have been removed from
public view, and so their justification for the behavior of this check requires
an account on their wiki to view.
