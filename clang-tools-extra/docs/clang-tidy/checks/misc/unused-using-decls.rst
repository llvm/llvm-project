.. title:: clang-tidy - misc-unused-using-decls

misc-unused-using-decls
=======================

Finds unused ``using`` declarations.

Unused ``using``` declarations in header files will not be diagnosed since these
using declarations are part of the header's public API. Allowed header file
extensions can be configured via the `HeaderFileExtensions` option (see below).

Example:

.. code-block:: c++

  // main.cpp
  namespace n { class C; }
  using n::C;  // Never actually used.

Options
-------

.. option:: HeaderFileExtensions

   Note: this option is deprecated, it will be removed in :program:`clang-tidy`
   version 19. Please use the global configuration option
   `HeaderFileExtensions`.

   A semicolon-separated list of filename extensions of header files (the filename
   extensions should not include "." prefix). Default is "h,hh,hpp,hxx".
   For extension-less header files, use an empty string or leave an
   empty string between "," if there are other filename extensions.
