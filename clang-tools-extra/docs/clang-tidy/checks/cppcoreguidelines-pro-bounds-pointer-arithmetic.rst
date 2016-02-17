.. title:: clang-tidy - cppcoreguidelines-pro-bounds-pointer-arithmetic

cppcoreguidelines-pro-bounds-pointer-arithmetic
===============================================

This check flags all usage of pointer arithmetic, because it could lead to an
invalid pointer. Subtraction of two pointers is not flagged by this check.

Pointers should only refer to single objects, and pointer arithmetic is fragile
and easy to get wrong. ``span<T>`` is a bounds-checked, safe type for accessing
arrays of data.

This rule is part of the "Bounds safety" profile of the C++ Core Guidelines, see
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Pro-bounds-arithmetic.
