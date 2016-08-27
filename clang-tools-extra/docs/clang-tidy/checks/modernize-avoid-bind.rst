.. title:: clang-tidy - modernize-avoid-bind

modernize-avoid-bind
====================

The check finds uses of ``std::bind`` and replaces simple uses with lambdas.
Lambdas will use value-capture where required.

Right now it only handles free functions, not member functions.

Given:

.. code-block:: c++

  int add(int x, int y) { return x + y; }

Then:

.. code-block:: c++

  void f() {
    int x = 2;
    auto clj = std::bind(add, x, _1);
  }

is replaced by:

.. code-block:: c++

  void f() {
    int x = 2;
    auto clj = [=](auto && arg1) { return add(x, arg1); };
  }

``std::bind`` can be hard to read and can result in larger object files and
binaries due to type information that will not be produced by equivalent
lambdas.
