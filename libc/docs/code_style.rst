.. _code_style:

===================
The libc code style
===================

For the large part, the libc project follows the general `coding standards of
the LLVM project <https://llvm.org/docs/CodingStandards.html>`_. The libc
project differs from that standard with respect to the naming style. The
differences are as follows:

#. **Non-const variables** - This includes function arguments, struct and
   class data members, non-const globals and local variables. They all use the
   ``snake_case`` style.
#. **const and constexpr variables** - They use the capitlized
   ``SNAKE_CASE`` irrespective of whether they are local or global.
#. **Function and methods** - They use the ``snake_case`` style like the
   non-const variables.
#. **Internal type names** - These are types which are interal to the libc
   implementation. They use the `CaptilizedCamelCase` style.
#. **Public names** - These are the names as prescribed by the standards and
   will follow the style as prescribed by the standards.
