.. title:: clang-tidy - cppcoreguidelines-avoid-capturing-lambda-coroutines

cppcoreguidelines-avoid-capturing-lambda-coroutines
===================================================

Warns if a capturing lambda is a coroutine. For example:

.. code-block:: c++

   int c;

   [c] () -> task { co_return; };
   [&] () -> task { int y = c; co_return; };
   [=] () -> task { int y = c; co_return; };

   struct s {
       void m() {
           [this] () -> task { co_return; };
       }
   };

All of the cases above will trigger the warning. However, implicit captures
do not trigger the warning unless the body of the lambda uses the capture.
For example, the following do not trigger the warning.

.. code-block:: c++

   int c;

   [&] () -> task { co_return; };
   [=] () -> task { co_return; };
