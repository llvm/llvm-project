.. title:: clang-tidy - performance-lost-std-move

performance-lost-std-move
=========================

Warns if copy constructor is used instead of ``std::move()``.

.. code-block:: c++

   void f(X);

   void g(X x) {
     f(x);  // warning: Could be std::move() [performance-lost-std-move]
   }
