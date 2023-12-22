.. title:: clang-tidy - readability-avoid-return-with-void-value

readability-avoid-return-with-void-value
========================================

Complains about statements returning expressions of type ``void``. It can be
confusing if a function returns an expression even though its return type is
``void``.

Example:

.. code-block::

   void g();
   void f() {
       // ...
       return g();
   }

In a long function body, the ``return`` statements suggests that the function
returns a value. However, ``return g();`` is combination of two statements that
should be written as

.. code-block::

   g();
   return;

to make clear that ``g()`` is called and immediately afterwards the function 
returns (nothing).
