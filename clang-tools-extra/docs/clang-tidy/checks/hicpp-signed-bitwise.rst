.. title:: clang-tidy - hicpp-signed-bitwise

hicpp-signed-bitwise
====================

Finds uses of bitwise operations on signed integer types, which may lead to 
undefined or implementation defined behaviour.

The according rule is defined in the `High Integrity C++ Standard, Section 5.6.1 <http://www.codingstandard.com/section/5-6-shift-operators/>`_.
