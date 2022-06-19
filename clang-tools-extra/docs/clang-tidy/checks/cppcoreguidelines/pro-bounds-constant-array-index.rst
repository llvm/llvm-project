.. title:: clang-tidy - cppcoreguidelines-pro-bounds-constant-array-index

cppcoreguidelines-pro-bounds-constant-array-index
=================================================

This check flags all array subscript expressions on static arrays and
``std::arrays`` that either do not have a constant integer expression index or
are out of bounds (for ``std::array``). For out-of-bounds checking of static
arrays, see the `-Warray-bounds` Clang diagnostic.

This rule is part of the "Bounds safety" profile of the C++ Core Guidelines, see
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Pro-bounds-arrayindex.

Optionally, this check can generate fixes using ``gsl::at`` for indexing.

Options
-------

.. option:: GslHeader

   The check can generate fixes after this option has been set to the name of
   the include file that contains ``gsl::at()``, e.g. `"gsl/gsl.h"`.

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.
