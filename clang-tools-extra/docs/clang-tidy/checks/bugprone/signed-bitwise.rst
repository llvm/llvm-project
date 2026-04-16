.. title:: clang-tidy - bugprone-signed-bitwise

bugprone-signed-bitwise
=======================

Finds uses of bitwise operations on signed integer types, which may lead to
undefined or implementation defined behavior.

Performing bitwise operations on signed integers can be confusing and may
lead to undefined or implementation dependent behavior. In particular, right
shift a signed integer is implementation dependent, while left shift a signed
integer may result in undefined behavior.

.. code-block:: c++

   int main(){
      int x = -4;
      int y = x >> 1; // y can be -2 or 2147483646
   }

Options
-------

.. option:: IgnorePositiveIntegerLiterals

   If this option is set to `true`, the check will not warn on bitwise
   operations with positive integer literals, e.g. ``~0``, ``2 << 1``, etc.
   Default value is `false`.
