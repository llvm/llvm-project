.. title:: clang-tidy - modernize-use-default

modernize-use-default
=====================

This check replaces default bodies of special member functions with ``=
default;``.  The explicitly defaulted function declarations enable more
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
  Copy-constructor, copy-assignment operator, move-constructor and
  move-assignment operator are not supported yet.
