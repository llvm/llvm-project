.. title:: clang-tidy - modernize-default-arg-braced-init

modernize-default-arg-braced-init
=================================

Replaces redundant non-explicit default constructor calls in default arguments with a braced
initializer list. This avoids unnecessarily repeating the type name in the
function declaration.

.. code:: c++

   void func(std::string s = std::string());
   void handle(Widget w = Widget());
   void process(Box b = Box());

   // transforms to:

   void func(std::string s = {});
   void handle(Widget w = {});
   void process(Box b = {});
