.. title:: clang-tidy - cppcoreguidelines-slicing

cppcoreguidelines-slicing
=========================

Flags slicing of member variables or vtable. Slicing happens when copying a
derived object into a base object: the members of the derived object (both
member variables and virtual member functions) will be discarded. This can be
misleading especially for member function slicing, for example:

.. code-block:: c++

  struct B { int a; virtual int f(); };
  struct D : B { int b; int f() override; };

  void use(B b) {  // Missing reference, intended?
    b.f();  // Calls B::f.
  }

  D d;
  use(d);  // Slice.

This check implements `ES.63
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#es63-dont-slice>`_
and `C.145
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c145-access-polymorphic-objects-through-pointers-and-references>`_
from the C++ Core Guidelines.
