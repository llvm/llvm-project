.. title:: clang-tidy - cppcoreguidelines-pro-bounds-array-to-pointer-decay

cppcoreguidelines-pro-bounds-array-to-pointer-decay
===================================================

This check flags all array to pointer decays.

Pointers should not be used as arrays. ``span<T>`` is a bounds-checked, safe
alternative to using pointers to access arrays.

This rule is part of the `Bounds safety (Bounds 3)
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Pro-bounds-decay>`_
profile from the C++ Core Guidelines.
