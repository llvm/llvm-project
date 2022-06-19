.. title:: clang-tidy - hicpp-signed-bitwise

hicpp-signed-bitwise
====================

Finds uses of bitwise operations on signed integer types, which may lead to
undefined or implementation defined behavior.

The according rule is defined in the `High Integrity C++ Standard, Section 5.6.1 <http://www.codingstandard.com/section/5-6-shift-operators/>`_.

Options
-------

.. option:: IgnorePositiveIntegerLiterals

   If this option is set to `true`, the check will not warn on bitwise operations with positive integer literals, e.g. `~0`, `2 << 1`, etc.
   Default value is `false`.
