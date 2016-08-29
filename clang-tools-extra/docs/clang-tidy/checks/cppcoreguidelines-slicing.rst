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

See the relevant C++ Core Guidelines sections for details:
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#es63-dont-slice
https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#c145-access-polymorphic-objects-through-pointers-and-references
