.. title:: clang-tidy - bugprone-default-operator-new-on-overaligned-type

bugprone-default-operator-new-on-overaligned-type
=================================================

Flags uses of default ``operator new`` where the type has extended
alignment (an alignment greater than the fundamental alignment).

The default ``operator new`` is guaranteed to provide the correct alignment
if the requested alignment is less or equal to the fundamental alignment.
Only cases are detected (by design) where the ``operator new`` is not
user-defined and is not a placement new (the reason is that in these cases we
assume that the user provided the correct memory allocation).

References
----------

This check corresponds to the CERT C++ Coding Standard rule
`MEM57-CPP. Avoid using default operator new for over-aligned types
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/MEM57-CPP.+Avoid+using+default+operator+new+for+over-aligned+types>`_.
