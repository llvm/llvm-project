.. title:: clang-tidy - performance-lost-std-move

performance-lost-std-move
=========================

Warns if copy constructor is used instead of ``std::move()`` and suggests a fix.
It honours cycles, lambdas, and unspecified call order in compound expressions.

.. code-block:: c++

   void f(X);

   void g(X x) {
     f(x);  // warning: Could be std::move() [performance-lost-std-move]
   }

It finds the last local variable usage, and if it is a copy, emits a warning.
The check is based on pure AST matching and doesn't take into account any
data flow information. Thus, it doesn't catch assign-after-copy cases.
Also it doesn't notice variable references "behind the scenes":

.. code-block:: c++

   void f(X);

   void g(X x) {
     auto &y = x;
     f(x);  // emits a warning...
     y.f();  // ...but it is still used
   }

Such rare cases should be silenced using `// NOLINT`.
