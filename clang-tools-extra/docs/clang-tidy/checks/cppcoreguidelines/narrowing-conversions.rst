.. title:: clang-tidy - cppcoreguidelines-narrowing-conversions

cppcoreguidelines-narrowing-conversions
=======================================

Checks for silent narrowing conversions, e.g: ``int i = 0; i += 0.1;``. While
the issue is obvious in this former example, it might not be so in the
following: ``void MyClass::f(double d) { int_member_ += d; }``.

This rule is part of the "Expressions and statements" profile of the C++ Core
Guidelines, corresponding to rule ES.46. See

https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#es46-avoid-lossy-narrowing-truncating-arithmetic-conversions.

We enforce only part of the guideline, more specifically, we flag narrowing conversions from:
 - an integer to a narrower integer (e.g. ``char`` to ``unsigned char``)
   if WarnOnIntegerNarrowingConversion Option is set,
 - an integer to a narrower floating-point (e.g. ``uint64_t`` to ``float``)
   if WarnOnIntegerToFloatingPointNarrowingConversion Option is set,
 - a floating-point to an integer (e.g. ``double`` to ``int``),
 - a floating-point to a narrower floating-point (e.g. ``double`` to ``float``)
   if WarnOnFloatingPointNarrowingConversion Option is set.

This check will flag:
 - All narrowing conversions that are not marked by an explicit cast (c-style or
   ``static_cast``). For example: ``int i = 0; i += 0.1;``,
   ``void f(int); f(0.1);``,
 - All applications of binary operators with a narrowing conversions.
   For example: ``int i; i+= 0.1;``.


Options
-------

.. option:: WarnOnIntegerNarrowingConversion

    When `true`, the check will warn on narrowing integer conversion
    (e.g. ``int`` to ``size_t``). `true` by default.

.. option:: WarnOnIntegerToFloatingPointNarrowingConversion

    When `true`, the check will warn on narrowing integer to floating-point
    conversion (e.g. ``size_t`` to ``double``). `true` by default.

.. option:: WarnOnFloatingPointNarrowingConversion

    When `true`, the check will warn on narrowing floating point conversion
    (e.g. ``double`` to ``float``). `true` by default.

.. option:: WarnWithinTemplateInstantiation

    When `true`, the check will warn on narrowing conversions within template
    instantiations. `false` by default.

.. option:: WarnOnEquivalentBitWidth

    When `true`, the check will warn on narrowing conversions that arise from
    casting between types of equivalent bit width. (e.g.
    `int n = uint(0);` or `long long n = double(0);`) `true` by default.

.. option:: IgnoreConversionFromTypes

   Narrowing conversions from any type in this semicolon-separated list will be
   ignored. This may be useful to weed out commonly occurring, but less commonly
   problematic assignments such as `int n = std::vector<char>().size();` or
   `int n = std::difference(it1, it2);`. The default list is empty, but one
   suggested list for a legacy codebase would be
   `size_t;ptrdiff_t;size_type;difference_type`.

.. option:: PedanticMode

    When `true`, the check will warn on assigning a floating point constant
    to an integer value even if the floating point value is exactly
    representable in the destination type (e.g. ``int i = 1.0;``).
    `false` by default.

FAQ
---

 - What does "narrowing conversion from 'int' to 'float'" mean?

An IEEE754 Floating Point number can represent all integer values in the range
[-2^PrecisionBits, 2^PrecisionBits] where PrecisionBits is the number of bits in
the mantissa.

For ``float`` this would be [-2^23, 2^23], where ``int`` can represent values in
the range [-2^31, 2^31-1].

 - What does "implementation-defined" mean?

You may have encountered messages like "narrowing conversion from 'unsigned int'
to signed type 'int' is implementation-defined".
The C/C++ standard does not mandate two's complement for signed integers, and so
the compiler is free to define what the semantics are for converting an unsigned
integer to signed integer. Clang's implementation uses the two's complement
format.
