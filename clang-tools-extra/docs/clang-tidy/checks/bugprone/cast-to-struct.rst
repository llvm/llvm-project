.. title:: clang-tidy - bugprone-cast-to-struct

bugprone-cast-to-struct
=======================

Finds casts from pointers to struct or scalar type to pointers to struct type.

Casts between pointers to different structs can be unsafe because it is possible
to access uninitialized or undefined data after the cast. There may be issues
with type compatibility or data alignment. Cast from a pointer to a scalar type
(which points often to an array or memory block) to a `struct` type pointer can
be unsafe for similar reasons. This check warns at casts from any non-`struct`
type to a `struct` type. No warning is produced at cast from type `void *` (this
is the usual way of allocating memory with `malloc`-like functions). It is
possible to specify additional types to ignore by the check. In addition,
`union` types are completely excluded from the check. The check does not take
into account type compatibility or data layout, only the names of the types.

.. code-block:: c

   void test1(char *p) {
     struct S1 *s;
     s = (struct S1 *)p; // warn: 'char *' is converted to 'struct S1 *'
   }

   void test2(struct S1 *p) {
     struct S2 *s;
     s = (struct S2 *)p; // warn: 'struct S1 *' is converted to 'struct S2 *'
   }

   void test3(void) {
     struct S1 *s;
     s = (struct S1 *)calloc(1, sizeof(struct S1)); // no warning
   }

Options
-------

.. option:: IgnoredFromTypes

   Semicolon-separated list of types for which the checker should not warn if
   encountered at cast source. Can contain regular expressions. The `*`
   character (for pointer type) is not needed in the type names.

.. option:: IgnoredToTypes

   Semicolon-separated list of types for which the checker should not warn if
   encountered at cast destination. Can contain regular expressions. The `*`
   character (for pointer type) is not needed in the type names.

.. option:: IgnoredFunctions

   List of function names from which the checker should produce no warnings. Can
   contain regular expressions.
