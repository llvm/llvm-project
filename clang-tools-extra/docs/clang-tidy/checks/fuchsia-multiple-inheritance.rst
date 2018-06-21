.. title:: clang-tidy - fuchsia-multiple-inheritance

fuchsia-multiple-inheritance
============================

Warns if a class inherits from multiple classes that are not pure virtual.

For example, declaring a class that inherits from multiple concrete classes is
disallowed:

.. code-block:: c++

  class Base_A {
  public:
    virtual int foo() { return 0; }
  };

  class Base_B {
  public:
    virtual int bar() { return 0; }
  };

  // Warning
  class Bad_Child1 : public Base_A, Base_B {};

A class that inherits from a pure virtual is allowed:

.. code-block:: c++

  class Interface_A {
  public:
    virtual int foo() = 0;
  };

  class Interface_B {
  public:
    virtual int bar() = 0;
  };

  // No warning
  class Good_Child1 : public Interface_A, Interface_B {
    virtual int foo() override { return 0; }
    virtual int bar() override { return 0; }
  };

See the features disallowed in Fuchsia at https://fuchsia.googlesource.com/zircon/+/master/docs/cxx.md
