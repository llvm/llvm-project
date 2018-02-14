.. title:: clang-tidy - bugprone-copy-constructor-init

bugprone-copy-constructor-init
==============================

Finds copy constructors where the constructor doesn't call
the copy constructor of the base class.

.. code-block:: c++

    class Copyable {
    public:
      Copyable() = default;
      Copyable(const Copyable &) = default;
    };
    class X2 : public Copyable {
      X2(const X2 &other) {} // Copyable(other) is missing
    };

Also finds copy constructors where the constructor of
the base class don't have parameter.

.. code-block:: c++

    class X4 : public Copyable {
      X4(const X4 &other) : Copyable() {} // other is missing
    };

The check also suggests a fix-its in some cases.
