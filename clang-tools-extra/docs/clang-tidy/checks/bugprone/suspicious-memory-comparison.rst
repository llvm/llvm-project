.. title:: clang-tidy - bugprone-suspicious-memory-comparison

bugprone-suspicious-memory-comparison
=====================================

Finds potentially incorrect calls to ``memcmp()`` based on properties of the
arguments. The following cases are covered:

**Case 1: Non-standard-layout type**

Comparing the object representations of non-standard-layout objects may not
properly compare the value representations.

**Case 2: Types with no unique object representation**

Objects with the same value may not have the same object representation.
This may be caused by padding or floating-point types.

See also:
`EXP42-C. Do not compare padding data
<https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/rules/expressions-exp/exp42-c/>`_
and
`FLP37-C. Do not use object representations to compare floating-point values
<https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/rules/floating-point-flp/flp37-c/>`_

This check is also related to and partially overlaps the CERT C++ Coding Standard rules
`OOP57-CPP. Prefer special member functions and overloaded operators to
C Standard Library functions
<https://cmu-sei.github.io/secure-coding-standards/sei-cert-cpp-coding-standard/rules/object-oriented-programming-oop/oop57-cpp/>`_
and
`EXP62-CPP. Do not access the bits of an object representation that are not
part of the object's value representation
<https://cmu-sei.github.io/secure-coding-standards/sei-cert-cpp-coding-standard/rules/expressions-exp/exp62-cpp/>`_

`cert-exp42-c` redirects here as an alias of this check.
