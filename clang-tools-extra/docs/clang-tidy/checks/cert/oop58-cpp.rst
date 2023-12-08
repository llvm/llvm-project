.. title:: clang-tidy - cert-mutating-copy

cert-oop58-cpp
==============

Finds assignments to the copied object and its direct or indirect members
in copy constructors and copy assignment operators.

This check corresponds to the CERT C Coding Standard rule
`OOP58-CPP. Copy operations must not mutate the source object
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP58-CPP.+Copy+operations+must+not+mutate+the+source+object>`_.
