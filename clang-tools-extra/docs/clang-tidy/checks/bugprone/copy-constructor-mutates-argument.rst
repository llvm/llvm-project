.. title:: clang-tidy - bugprone-copy-constructor-mutates-argument

bugprone-copy-constructor-mutates-argument
==========================================

Finds assignments to the copied object and its direct or indirect members
in copy constructors and copy assignment operators.

This check corresponds to the CERT C Coding Standard rule
`OOP58-CPP. Copy operations must not mutate the source object
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP58-CPP.+Copy+operations+must+not+mutate+the+source+object>`_.
