.. title:: clang-tidy - bugprone-bool-pointer-implicit-conversion

bugprone-bool-pointer-implicit-conversion
=========================================

Checks for conditions based on implicit conversion from a ``bool`` pointer to
``bool``.

Example:

.. code-block:: c++

  bool *p;
  if (p) {
    // Never used in a pointer-specific way.
  }
