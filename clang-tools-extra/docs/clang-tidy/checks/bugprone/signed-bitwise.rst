.. title:: clang-tidy - bugprone-signed-bitwise

bugprone-signed-bitwise
=======================

Finds uses of bitwise operations on signed integer types, which may lead to
undefined or implementation defined behavior.

Options
-------

.. option:: IgnorePositiveIntegerLiterals

   If this option is set to `true`, the check will not warn on bitwise operations with positive integer literals, e.g. `~0`, `2 << 1`, etc.
   Default value is `false`.
