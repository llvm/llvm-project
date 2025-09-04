.. title:: clang-tidy - bugprone-cast-to-struct

bugprone-cast-to-struct
=======================

Finds casts from pointers to struct or scalar type to pointers to struct type.

Casts between pointers to different structs can be unsafe because it is possible
to access uninitialized or undefined data after the cast. Cast from a
scalar-type pointer (which points often to an array or memory block) to a
``struct`` type pointer can be unsafe for similar reasons. This check warns at
pointer casts from any non-struct type to a struct type. No warning is produced
at cast from type ``void *`` (this is the usual way of allocating memory with
``malloc``-like functions) and ``char *`` types (which are used often as
pointers into data buffers). In addition, ``union`` types are excluded from the
check. It is possible to specify additional types to ignore. The check does not
take into account type compatibility or data layout, only the names of the
types.

.. code-block:: c

   void test1(int *p) {
     struct S1 *s;
     s = (struct S1 *)p; // warn: 'int *' is converted to 'struct S1 *'
   }

   void test2(struct S1 *p) {
     struct S2 *s;
     s = (struct S2 *)p; // warn: 'struct S1 *' is converted to 'struct S2 *'
   }

   void test3(void) {
     struct S1 *s;
     s = (struct S1 *)calloc(1, sizeof(struct S1)); // no warning
   }

Limitations
-----------

The check does not run on `C++` code.

C-style casts are discouraged in C++ and should be converted to more type-safe
casts. The ``reinterpreted_cast`` is used for the most unsafe cases and
indicates by itself a potentially dangerous operation. Additionally, inheritance
and dynamic types would make such a check less useful.

Options
-------

.. option:: IgnoredCasts

   Can contain a semicolon-separated list of type names that specify cast
   types to ignore. The list should contain pairs of type names in a way that
   the first type is the "from" type, the second is the "to" type in a cast
   expression. The types in a pair and the pairs itself are separated by
   `;` characters. The parts between `;` characters are matched as regular
   expressions over the whole type name. For example
   `struct S1 \*;struct T1 \*;short \*;struct T1 \*` specifies that the check
   does not produce warning for casts from ``struct S1 *`` to ``struct T1 *``
   and casts from ``short *`` to ``struct T1 *`` (the `*` character needs to be
   escaped). The type name in the cast expression is matched without resolution
   of ``typedef`` types.
   
   Default value of the option is an empty list.
