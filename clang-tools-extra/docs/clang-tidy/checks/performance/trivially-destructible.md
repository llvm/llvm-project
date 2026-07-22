.. title:: clang-tidy - performance-trivially-destructible

performance-trivially-destructible
==================================

Finds types that could be made trivially-destructible by removing out-of-line
defaulted destructor declarations.

.. code-block:: c++

   struct A: TrivialType {
     ~A(); // Makes A non-trivially-destructible.
     TrivialType trivial_fields;
   };
   A::~A() = default;
