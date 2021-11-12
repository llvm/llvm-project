.. title:: clang-tidy - google-global-names-in-headers

google-global-names-in-headers
==============================

Flag global namespace pollution in header files. Right now it only triggers on
``using`` declarations and directives.

The relevant style guide section is
https://google.github.io/styleguide/cppguide.html#Namespaces.

Options
-------

.. option:: HeaderFileExtensions

   A comma-separated list of filename extensions of header files (the filename
   extensions should not contain "." prefix). Default is "h".
   For header files without an extension, use an empty string (if there are no
   other desired extensions) or leave an empty element in the list. e.g.,
   "h,hh,hpp,hxx," (note the trailing comma).
