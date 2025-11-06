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

.. option::  IgnoredRegex

   A regular expression for the file name to be ignored by the check. Default
   is empty string.
