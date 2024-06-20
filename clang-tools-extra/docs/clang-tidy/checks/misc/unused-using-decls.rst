.. title:: clang-tidy - misc-unused-using-decls

misc-unused-using-decls
=======================

Finds unused ``using`` declarations.

Unused ``using``` declarations in header files will not be diagnosed since these
using declarations are part of the header's public API. Allowed header file
extensions can be configured via the global option `HeaderFileExtensions`.

Example:

.. code-block:: c++

  // main.cpp
  namespace n { class C; }
  using n::C;  // Never actually used.
