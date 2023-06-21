.. title:: clang-tidy - llvmlibc-inline-function-decl

llvmlibc-inline-function-decl
=============================

Checks that all implicit and explicit inline functions in header files are
tagged with the ``LIBC_INLINE`` macro. See the `libc style guide
<https://libc.llvm.org/dev/code_style.html>`_ for more information about this macro.
