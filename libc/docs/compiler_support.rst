.. _compiler_support:

================
Compiler Support
================

``LLVM libc`` compiles from both ``Clang`` and ``GCC`` but for maximum
performance we recommend using ``Clang``.

Indeed, some memory function implementations rely on `compiler intrinsics`__
that are not currently available in ``GCC``.
As such we cannot guarantee optimal performance for these functions.

.. __: https://clang.llvm.org/docs/LanguageExtensions.html#guaranteed-inlined-copy

For platforms where only ``GCC`` is natively available but maximum performance
is required it is possible to bootstrap ``Clang`` with ``GCC`` and then use
``Clang`` to build the '`libc``" project.

IMPORTANT NOTE: There is currently an issue when doing an overlay build in
release mode with GCC. If you want to build with GCC, please either build in
debug mode or use fullbuild mode. Otherwise, please use clang.

Minimum supported versions
==========================

 - ``Clang 11``
 - ``GCC 12.2``
