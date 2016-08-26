.. title:: clang-tidy - llvm-header-guard

llvm-header-guard
=================

Finds and fixes header guards that do not adhere to LLVM style.

Options
-------

.. option:: HeaderFileExtensions

   A comma-separated list of filename extensions of header files (The filename
   extension should not contain "." prefix). Default value is ",h,hh,hpp,hxx".
   For extension-less header files, using an empty string or leaving an empty
   string between "," if there are other filename extensions.
