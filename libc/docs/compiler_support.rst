.. _compiler_support:

================
Compiler Support
================

As of now only ``Clang`` is fully supported.

We are in the process of supporting ``GCC`` but some memory function implementations rely on `compiler intrinsics`__ that are not currently available in ``GCC``.
As such we cannot guarantee optimal performance for these functions.

.. __: https://clang.llvm.org/docs/LanguageExtensions.html#guaranteed-inlined-copy

Minimum version
===============

 - ``Clang 11``
