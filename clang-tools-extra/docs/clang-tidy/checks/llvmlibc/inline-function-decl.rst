.. title:: clang-tidy - llvmlibc-inline-function-decl

llvmlibc-inline-function-decl
=============================

Checks that all implicit and explicit inline functions in header files are
tagged with the ``LIBC_INLINE`` macro. See the `libc style guide
<https://libc.llvm.org/code_style.html>`_ for more information about this macro.

Options
-------

.. option:: HeaderFileExtensions

   A comma-separated list of filename extensions of header files (the filename
   extensions should not include "." prefix). Default is "h,hh,hpp,hxx".
   For header files without an extension, use an empty string (if there are no
   other desired extensions) or leave an empty element in the list. E.g.,
   "h,hh,hpp,hxx," (note the trailing comma).
