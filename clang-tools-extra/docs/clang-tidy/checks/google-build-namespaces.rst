.. title:: clang-tidy - google-build-namespaces

google-build-namespaces
=======================

`cert-dcl59-cpp` redirects here as an alias for this check.
`fuchsia-header-anon-namespaces` redirects here as an alias for this check.

Finds anonymous namespaces in headers.

https://google.github.io/styleguide/cppguide.html#Namespaces

Corresponding cpplint.py check name: `build/namespaces`.

Options
-------

.. option:: HeaderFileExtensions

   A comma-separated list of filename extensions of header files (the filename
   extensions should not include "." prefix). Default is "h,hh,hpp,hxx".
   For header files without an extension, use an empty string (if there are no
   other desired extensions) or leave an empty element in the list. e.g.,
   "h,hh,hpp,hxx," (note the trailing comma).
