.. title:: clang-tidy - bugprone-raw-memory-call-on-non-trivial-type

bugprone-raw-memory-call-on-non-trivial-type
============================================

Flags use of the C standard library functions ``memset``, ``memcpy`` and
``memcmp`` and similar derivatives on non-trivial types.

The check will detect the following functions: ``memset``, ``std::memset``,
``std::memcpy``, ``memcpy``, ``std::memmove``, ``memmove``, ``std::strcpy``,
``strcpy``, ``memccpy``, ``stpncpy``, ``strncpy``, ``std::memcmp``, ``memcmp``,
``std::strcmp``, ``strcmp``, ``strncmp``.

Options
-------

.. option:: MemSetNames

   Specify extra functions to flag that act similarly to ``memset``. Specify
   names in a semicolon-delimited list. Default is an empty string.

.. option:: MemCpyNames

   Specify extra functions to flag that act similarly to ``memcpy``. Specify
   names in a semicolon-delimited list. Default is an empty string.

.. option:: MemCmpNames

   Specify extra functions to flag that act similarly to ``memcmp``. Specify
   names in a semicolon-delimited list. Default is an empty string.

This check corresponds to the CERT C++ Coding Standard rule
`OOP57-CPP. Prefer special member functions and overloaded operators to C
Standard Library functions
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP57-CPP.+Prefer+special+member+functions+and+overloaded+operators+to+C+Standard+Library+functions>`_.
