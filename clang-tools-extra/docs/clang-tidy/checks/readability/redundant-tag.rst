.. title:: clang-tidy - readability-redundant-tag

readability-redundant-tag
=========================

Finds redundant uses of the ``class``, ``struct``, ``union``, and ``enum``
keywords in C++ declarations and provides fix-it hints to remove them.

In C++, elaborated type specifiers are unnecessary when the type name is
already unambiguous.

For example:

.. code-block:: c++

  struct S {};

  void f() {
    struct S s;
  }

becomes:

.. code-block:: c++

  struct S {};

  void f() {
    S s;
  }

The check does not diagnose cases where removing the keyword would change name
lookup semantics. For example:

.. code-block:: c++

  struct Hidden {} Hidden;

  void f() {
    struct Hidden h;
  }

Removing ``struct`` would cause ``Hidden`` to refer to the variable rather
than the type.

Similarly:

.. code-block:: c++

  struct Foo {};

  void Foo();

  void f() {
    struct Foo x;
  }

Removing ``struct`` would cause ``Foo`` to refer to the function instead of
the type.
