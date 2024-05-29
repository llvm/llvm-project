.. title:: clang-tidy - bugprone-pointer-arithmetic-on-polymorphic-object

bugprone-pointer-arithmetic-on-polymorphic-object
=================================================

Finds pointer arithmetic performed on classes that declare a virtual function.

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
    // warning: pointer arithmetic on class that declares a virtual function,
    //          which can result in undefined behavior if the pointee is a
    //          different class

    delete[] static_cast<Derived*>(b);
  }

This check corresponds to the SEI Cert rule `CTR56-CPP: Do not use pointer arithmetic on polymorphic objects <https://wiki.sei.cmu.edu/confluence/display/cplusplus/CTR56-CPP.+Do+not+use+pointer+arithmetic+on+polymorphic+objects>`_.

Options
-------

.. option:: MatchInheritedVirtualFunctions

  When ``true``, all classes with a virtual function are considered,
  even if the function is inherited.
  Classes that do not declare a new virtual function are excluded
  by default, as they make up the majority of false positives.

  .. code-block:: c++
  
    void bar() {
      Base *b = new Base[10];
      b += 1; // warning, as Base declares a virtual destructor
  
      delete[] b;
  
      Derived *d = new Derived[10]; // Derived overrides the destructor, and
                                    // declares no other virtual functions
      d += 1; // warning only if MatchVirtualDeclarationsOnly is set to true
  
      delete[] d;
    }
