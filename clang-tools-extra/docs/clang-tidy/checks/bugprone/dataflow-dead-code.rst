.. title:: clang-tidy - bugprone-dataflow-dead-code

bugprone-dataflow-dead-code
===========================

*Note*: This check uses a flow-sensitive static analysis to produce its
results. Therefore, it may be more resource intensive (RAM, CPU) than the
average clang-tidy check.

Finds instances of always-true and always-false conditions in branch statements.

.. code-block:: c++

   void f(bool a, bool b) {
     if (a) {
       return;
     } else if (a == b) {
       if (b) { // warning: dead code - branching condition is always false
         return;
       }
     }
   }

Notes
-----

True and false literals
-----------------------

Since macro and template code commonly uses always-true and always-false loops,
the literals ``true`` and ``false`` are excluded from being matched outright.
Assertion statements are a common example.

.. code-block:: c++

   // common way to define asserts in libraries
   #define assert(x) do {} while(false)

   void f(int *param) {
     assert(param); // no-warning, even though while(false) is always false
   }

C++ class support
-----------------

Support for C++ datastructures is limited due to framework limitations.
Calling non-const member functions of a class do not invalidate member variable
values.

.. code-block:: c++

   struct S {
     bool a;
     void change_a() { a = random_bool(); }
   };

   void f(S s) {
     if (s.a) {
       return;
     }
     s.change_a();
     if (s.a) {} // false-positive: condition is always false
   }

Marking of unexpected values
----------------------------

Due to framework limitations, the check currently utilizes a mark-and-check
approach. First it marks all loop condition values, then it checks whether the
value is always true or not. This can lead to the same value showing as
always-true in an unexpected place, or in an unexpected expression.

.. code-block:: c++

   void f(int a, int b) {
     if (a == b) {
       f(a == b); // unexpected-warning: condition is always true
     }
   }
