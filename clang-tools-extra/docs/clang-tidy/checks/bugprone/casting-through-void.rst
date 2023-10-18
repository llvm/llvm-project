.. title:: clang-tidy - bugprone-casting-through-void

bugprone-casting-through-void
=============================

A check detects unsafe or redundant two-step casting operations involving ``void*``.

Two-step type conversions via void* are discouraged for several reasons.

1. They obscure code and impede its understandability, complicating maintenance.
2. These conversions bypass valuable compiler support, erasing warnings related
   to pointer alignment. It may violate strict aliasing rule and leading to
   undefined behavior.
3. In scenarios involving multiple inheritance, ambiguity and unexpected outcomes
   can arise due to the loss of type information, posing runtime issues.

In summary, avoiding two-step type conversions through void* ensures clearer code,
maintains essential compiler warnings, and prevents ambiguity and potential runtime
errors, particularly in complex inheritance scenarios.

Examples:

.. code-block:: c++
   using IntegerPointer = int *;
   double *ptr;

   static_cast<IntegerPointer>(static_cast<void *>(ptr)); // WRONG
   reinterpret_cast<IntegerPointer>(reinterpret_cast<void *>(ptr)); // WRONG
   (IntegerPointer)(void *)ptr; // WRONG
   IntegerPointer(static_cast<void *>(ptr)); // WRONG

   __bultin_bit_cast(IntegerPointer, static_cast<void *>(ptr)) // GOOD, __builtin_bit_cast is used to implement std::bit_cast
