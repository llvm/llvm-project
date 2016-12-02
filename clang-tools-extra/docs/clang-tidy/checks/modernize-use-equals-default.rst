.. title:: clang-tidy - modernize-use-equals-default

modernize-use-equals-default
============================

This check replaces default bodies of special member functions with ``=
default;``. The explicitly defaulted function declarations enable more
opportunities in optimization, because the compiler might treat explicitly
defaulted functions as trivial.

.. code-block:: c++

  struct A {
    A() {}
    ~A();
  };
  A::~A() {}

  // becomes

  struct A {
    A() = default;
    ~A();
  };
  A::~A() = default;

.. note::
  Move-constructor and move-assignment operator are not supported yet.
