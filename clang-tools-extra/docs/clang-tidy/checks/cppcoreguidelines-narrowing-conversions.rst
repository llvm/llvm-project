.. title:: clang-tidy - cppcoreguidelines-narrowing-conversions

cppcoreguidelines-narrowing-conversions
=======================================

Checks for silent narrowing conversions, e.g: ``int i = 0; i += 0.1;``. While
the issue is obvious in this former example, it might not be so in the
following: ``void MyClass::f(double d) { int_member_ += d; }``.

This rule is part of the "Expressions and statements" profile of the C++ Core
Guidelines, corresponding to rule ES.46. See

https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-narrowing.

We enforce only part of the guideline, more specifically, we flag:
 - All floating-point to integer conversions that are not marked by an explicit
   cast (c-style or ``static_cast``). For example: ``int i = 0; i += 0.1;``,
   ``void f(int); f(0.1);``,
 - All applications of binary operators where the left-hand-side is an integer
   and the right-hand-size is a floating-point. For example:
   ``int i; i+= 0.1;``.

