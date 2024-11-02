.. title:: clang-tidy - misc-unused-using-decls

misc-unused-using-decls
=======================

Finds unused ``using`` declarations.

Example:

.. code-block:: c++

  namespace n { class C; }
  using n::C;  // Never actually used.
