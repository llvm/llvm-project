.. title:: clang-tidy - cert-oop57-cpp

cert-oop57-cpp
==============

  Flags use of the `C` standard library functions ``memset``, ``memcpy`` and
  ``memcmp`` and similar derivatives on non-trivial types.

Options
-------

.. option:: MemSetNames

   Specify extra functions to flag that act similarily to ``memset``.
   Specify names in a semicolon delimited list.
   Default is an empty string.
   The check will detect the following functions:
   `memset`, `std::memset`.

.. option:: MemCpyNames

   Specify extra functions to flag that act similarily to ``memcpy``.
   Specify names in a semicolon delimited list.
   Default is an empty string.
   The check will detect the following functions:
   `std::memcpy`, `memcpy`, `std::memmove`, `memmove`, `std::strcpy`,
   `strcpy`, `memccpy`, `stpncpy`, `strncpy`.

.. option:: MemCmpNames

   Specify extra functions to flag that act similarily to ``memcmp``.
   Specify names in a semicolon delimited list.
   Default is an empty string.
   The check will detect the following functions:
   `std::memcmp`, `memcmp`, `std::strcmp`, `strcmp`, `strncmp`.

This check corresponds to the CERT C++ Coding Standard rule
`OOP57-CPP. Prefer special member functions and overloaded operators to C 
Standard Library functions
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP57-CPP.+Prefer+special+member+functions+and+overloaded+operators+to+C+Standard+Library+functions>`_.
