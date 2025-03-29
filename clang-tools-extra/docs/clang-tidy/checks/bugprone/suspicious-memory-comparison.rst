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
<https://wiki.sei.cmu.edu/confluence/display/c/EXP42-C.+Do+not+compare+padding+data>`_
and
`FLP37-C. Do not use object representations to compare floating-point values
<https://wiki.sei.cmu.edu/confluence/display/c/FLP37-C.+Do+not+use+object+representations+to+compare+floating-point+values>`_

This check is also related to and partially overlaps the CERT C++ Coding Standard rules
`OOP57-CPP. Prefer special member functions and overloaded operators to C Standard Library functions
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP57-CPP.+Prefer+special+member+functions+and+overloaded+operators+to+C+Standard+Library+functions>`_
and
`EXP62-CPP. Do not access the bits of an object representation that are not part of the object's value representation
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/EXP62-CPP.+Do+not+access+the+bits+of+an+object+representation+that+are+not+part+of+the+object%27s+value+representation>`_

`cert-exp42-c` redirects here as an alias of this check.
