.. title:: clang-tidy - misc-forbid-non-virtual-base-dtor

misc-forbid-non-virtual-base-dtor
=================================

Warns when a class or struct publicly inherits from a base class or struct
whose destructor is neither virtual nor protected, and the derived class adds
data members. This pattern causes resource leaks when the derived object is
deleted through a base class pointer, because the derived destructor is never
called.

Examples
--------

The following code will trigger a warning:

.. code-block:: c++

  class Base {};  // non-virtual destructor

  class Derived : public Base {  // warning: class 'Derived' inherits from
      int data;                  // 'Base' which has a non-virtual destructor
  };

  Base *b = new Derived();
  delete b;  // leaks Derived::data —> Base::~Base() is called, not ~Derived()

The following patterns are safe and will **not** trigger a warning:

.. code-block:: c++

  class Base1 {
  public:
      virtual ~Base1() {}
  };
  class Derived1 : public Base1 { int data; };

destructor (prevents delete-through-base)
  class Base2 {
  protected:
      ~Base2() {}
  };
  class Derived2 : public Base2 { int data; };

  class Base3 {};
  class Derived3 : public Base3 {};  // OK

  class Base4 {};
  class Derived4 : private Base4 { int data; };
