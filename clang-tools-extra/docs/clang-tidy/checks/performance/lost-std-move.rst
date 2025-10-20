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

Also it does notice variable references "behind the scenes":

.. code-block:: c++

   void f(X);

   void g(X x) {
     auto &y = x;
     f(x);  // does not emit a warning
     y.f();  // because x is still used via y
   }

If you want to ignore assigns to reference variables, set :option:`StrictMode`
to `true`.


Options
-------

.. option:: StrictMode

   A variable ``X`` can be referenced by another variable ``R``. In this case
   the last variable usage might be not from ``X``, but from ``R``. It is quite
   difficult to find in a large function, so if the plugin sees some ``R``
   references ``X``, then it will not emit a warning for ``X`` not to provoke
   false positive. If you're sure that such references don't extend ``X``
   lifetime and ready to handle possible false positives, then set
   :option:`StrictMode` to `true`. Default is `false`.
