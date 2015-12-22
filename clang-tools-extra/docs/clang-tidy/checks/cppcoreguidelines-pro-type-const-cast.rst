.. title:: clang-tidy - cppcoreguidelines-pro-type-const-cast

cppcoreguidelines-pro-type-const-cast
=====================================

This check flags all uses of const_cast in C++ code.

Modifying a variable that was declared const is undefined behavior, even with const_cast.

This rule is part of the "Type safety" profile of the C++ Core Guidelines, see
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#-type3-dont-use-const_cast-to-cast-away-const-ie-at-all.
