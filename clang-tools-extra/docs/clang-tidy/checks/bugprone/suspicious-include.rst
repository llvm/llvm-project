.. title:: clang-tidy - bugprone-suspicious-include

bugprone-suspicious-include
===========================

The check detects various cases when an include refers to what appears to be an
implementation file, which often leads to hard-to-track-down ODR violations.

Examples:

.. code-block:: c++

  #include "Dinosaur.hpp"     // OK, .hpp files tend not to have definitions.
  #include "Pterodactyl.h"    // OK, .h files tend not to have definitions.
  #include "Velociraptor.cpp" // Warning, filename is suspicious.
  #include_next <stdio.c>     // Warning, filename is suspicious.

Options
-------
.. option:: HeaderFileExtensions

   Note: this option is deprecated, it will be removed in :program:`clang-tidy`
   version 18. Please use the global configuration option
   `HeaderFileExtensions`.

   Default value: ``";h;hh;hpp;hxx"``
   A semicolon-separated list of filename extensions of header files (the
   filename extensions should not contain a "." prefix). For extension-less
   header files, use an empty string or leave an empty string between ";"
   if there are other filename extensions.

.. option:: ImplementationFileExtensions

   Note: this option is deprecated, it will be removed in :program:`clang-tidy`.
   version 18. Please use the global configuration option
   `ImplementationFileExtensions`.

   Default value: ``"c;cc;cpp;cxx"``
   Likewise, a semicolon-separated list of filename extensions of
   implementation files.
