.. title:: clang-tidy - bugprone-pointer-arithmetic-on-polymorphic-object

bugprone-pointer-arithmetic-on-polymorphic-object
=================================================

Finds pointer arithmetic performed on classes that contain a virtual function.

Pointer arithmetic on polymorphic objects where the pointer's static type is
different from its dynamic type is undefined behavior, as the two types could
have different sizes, and thus the vtable pointer could point to an
invalid address.

Finding pointers where the static type contains a virtual member function is a
good heuristic, as the pointer is likely to point to a different,
derived object.

Example:

.. code-block:: c++

  struct Base {
    virtual ~Base();
    int i;
  };
  
  struct Derived : public Base {};
  
  // Function that takes a pointer and performs pointer arithmetic
  void foo(Base* b) {
    b += 1;
    // warning: pointer arithmetic on class that declares a virtual function can
    // result in undefined behavior if the dynamic type differs from the
    // pointer type
  }
  
  void bar() {
    Derived d[10];  // Array of derived objects
    foo(d);         // Passing Derived pointer to foo(), which performs arithmetic
  }

  // Another example showing array access with polymorphic objects.
  int bar(const Derived d[]) {
    return d[1].i; // warning due to pointer arithmetic on polymorphic object
  }

  // Making Derived final suppresses the warning
  struct FinalDerived final : public Base {};

  int baz(const FinalDerived d[]) {
    return d[1].i; // no warning as FinalDerived is final
  }

Options
-------

.. option:: IgnoreInheritedVirtualFunctions

  When `true`, objects that only inherit a virtual function are not checked.
  Classes that do not declare a new virtual function are excluded
  by default, as they make up the majority of false positives.
  Default: `false`.

  .. code-block:: c++
  
    void bar() {
      Base *b = new Base[10];
      b += 1; // warning, as Base declares a virtual destructor

      delete[] b;

      Derived *d = new Derived[10]; // Derived overrides the destructor, and
                                    // declares no other virtual functions
      d += 1; // warning only if IgnoreVirtualDeclarationsOnly is set to false

      delete[] d;

      FinalDerived *f = new FinalDerived[10];
      f += 1; // no warning, FinalDerived is final and cannot be further derived

      delete[] f;
    }

References
----------

This check corresponds to the SEI Cert rule
`CTR56-CPP. Do not use pointer arithmetic on polymorphic objects
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/CTR56-CPP.+Do+not+use+pointer+arithmetic+on+polymorphic+objects>`_.
