.. title:: clang-tidy - bugprone-casting-through-void

bugprone-casting-through-void
=============================

Detects unsafe or redundant two-step casting operations involving ``void*``,
which is equivalent to ``reinterpret_cast`` as per the
`C++ Standard <https://eel.is/c++draft/expr.reinterpret.cast#7>`_.

Two-step type conversions via ``void*`` are discouraged for several reasons.

- They obscure code and impede its understandability, complicating maintenance.
- These conversions bypass valuable compiler support, erasing warnings related
  to pointer alignment. It may violate strict aliasing rule and leading to
  undefined behavior.
- In scenarios involving multiple inheritance, ambiguity and unexpected outcomes
  can arise due to the loss of type information, posing runtime issues.

In summary, avoiding two-step type conversions through ``void*`` ensures clearer code,
maintains essential compiler warnings, and prevents ambiguity and potential runtime
errors, particularly in complex inheritance scenarios. If such a cast is wanted,
it shall be done via ``reinterpret_cast``, to express the intent more clearly.

Note: it is expected that, after applying the suggested fix and using
``reinterpret_cast``, the check :doc:`cppcoreguidelines-pro-type-reinterpret-cast
<../cppcoreguidelines/pro-type-reinterpret-cast>` will emit a warning.
This is intentional: ``reinterpret_cast`` is a dangerous operation that can
easily break the strict aliasing rules when dereferencing the casted pointer,
invoking Undefined Behavior. The warning is there to prompt users to carefuly
analyze whether the usage of ``reinterpret_cast`` is safe, in which case the
warning may be suppressed.

Examples:

.. code-block:: c++

   using IntegerPointer = int *;
   double *ptr;

   static_cast<IntegerPointer>(static_cast<void *>(ptr)); // WRONG
   reinterpret_cast<IntegerPointer>(reinterpret_cast<void *>(ptr)); // WRONG
   (IntegerPointer)(void *)ptr; // WRONG
   IntegerPointer(static_cast<void *>(ptr)); // WRONG

   reinterpret_cast<IntegerPointer>(ptr); // OK, clearly expresses intent.
                                          // NOTE: dereferencing this pointer violates
                                          // the strict aliasing rules, invoking
                                          // Undefined Behavior.
