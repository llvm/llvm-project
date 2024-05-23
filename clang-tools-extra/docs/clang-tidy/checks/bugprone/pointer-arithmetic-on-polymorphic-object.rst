.. title:: clang-tidy - bugprone-pointer-arithmetic-on-polymorphic-object

bugprone-pointer-arithmetic-on-polymorphic-object
=================================================

Warn if pointer arithmetic is performed on a class that declares a
virtual function.

Pointer arithmetic on polymorphic objects where the pointer's static type is 
different from its dynamic type is undefined behavior.
Finding pointers where the static type contains a virtual member function is a
good heuristic, as the pointer is likely to point to a different, derived class.

Example:

.. code-block:: c++

  struct Base {
    virtual void ~Base();
  };

  struct Derived : public Base {};

  void foo() {
    Base *b = new Derived[10];

    b += 1;
    // warning: pointer arithmetic on class that declares a virtual function, undefined behavior if the pointee is a different class

    delete[] static_cast<Derived*>(b);
  }

Classes that do not declare a new virtual function are excluded,
as they make up the majority of false positives.

.. code-block:: c++

  void bar() {
    Base *b = new Base[10];
    b += 1; // warning, as Base has a virtual destructor

    delete[] b;

    Derived *d = new Derived[10];
    d += 1; // no warning, as Derived overrides the destructor

    delete[] d;
  }

This check corresponds to the SEI Cert rule `CTR56-CPP: Do not use pointer arithmetic on polymorphic objects <https://wiki.sei.cmu.edu/confluence/display/cplusplus/CTR56-CPP.+Do+not+use+pointer+arithmetic+on+polymorphic+objects>`_.
