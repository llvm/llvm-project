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
   extensions should not contain "." prefix). "h" by default. For extension-less
   header files, using an empty string or leaving an empty string between ","
   if there are other filename extensions.
