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
``malloc``-like functions). In addition, ``union`` types are excluded from the
check. It is possible to specify additional types to ignore. The check does not
take into account type compatibility or data layout, only the names of the
types.

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

Limitations
-----------

The check does run only on `C` code.

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
   `;` characters. For example `char;Type1;char;Type2` specifies that the
   check does not produce warning for casts from ``char *`` to ``Type1 *`` and
   casts from ``char *`` to ``Type2 *``. The list entries can be regular
   expressions. The type name in the cast expression is matched without
   resolution of type aliases like ``typedef``. Default value is empty list.
   (Casts from ``void *`` are ignored always regardless of this list.)
