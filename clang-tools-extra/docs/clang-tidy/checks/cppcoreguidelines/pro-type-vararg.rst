.. title:: clang-tidy - cppcoreguidelines-pro-type-vararg

cppcoreguidelines-pro-type-vararg
=================================

This check flags all calls to c-style vararg functions and all use of
``va_arg``.

To allow for SFINAE use of vararg functions, a call is not flagged if a literal
0 is passed as the only vararg argument or function is used in unevaluated
context.

Passing to varargs assumes the correct type will be read. This is fragile
because it cannot generally be enforced to be safe in the language and so relies
on programmer discipline to get it right.

This rule is part of the `Type safety (Type.8)
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Pro-type-varargs>`_
profile from the C++ Core Guidelines.
