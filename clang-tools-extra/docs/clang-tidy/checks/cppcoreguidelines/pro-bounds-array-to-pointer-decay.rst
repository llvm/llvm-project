.. title:: clang-tidy - cppcoreguidelines-pro-bounds-array-to-pointer-decay

cppcoreguidelines-pro-bounds-array-to-pointer-decay
===================================================

This check flags all array to pointer decays.

Pointers should not be used as arrays. ``span<T>`` is a bounds-checked, safe
alternative to using pointers to access arrays.

This rule is part of the "Bounds safety" profile of the C++ Core Guidelines, see
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Pro-bounds-decay.
