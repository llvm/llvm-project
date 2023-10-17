.. title:: clang-tidy - llvm-header-guard

llvm-header-guard
=================

Finds and fixes header guards that do not adhere to LLVM style.

Options
-------

.. option:: HeaderFileExtensions

   Note: this option is deprecated, it will be removed in :program:`clang-tidy`
   version 19. Please use the global configuration option
   `HeaderFileExtensions`.

   A comma-separated list of filename extensions of header files (the filename
   extensions should not include "." prefix). Default is "h,hh,hpp,hxx".
   For header files without an extension, use an empty string (if there are no
   other desired extensions) or leave an empty element in the list. E.g.,
   "h,hh,hpp,hxx," (note the trailing comma).
