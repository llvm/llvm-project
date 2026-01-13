.. title:: clang-tidy - bugprone-derived-method-shadowing-base-method

bugprone-derived-method-shadowing-base-method
=============================================

Finds derived class methods that shadow a (non-virtual) base class method.

In order to be considered "shadowing", methods must have the same signature
(i.e. the same name, same number of parameters, same parameter types, etc).
Only checks public, non-templated methods.

The below example is bugprone because consumers of the ``Derived`` class will
expect the ``reset`` method to do the work of ``Base::reset()`` in addition to
extra work required to reset the ``Derived`` class.  Common fixes include:

- Making the ``reset`` method polymorphic
- Re-naming ``Derived::reset`` if it's not meant to intersect with
  ``Base::reset``
- Using ``using Base::reset`` to change the access specifier

This is also a violation of the Liskov Substitution Principle.

.. code-block:: c++

  struct Base {
    void reset() {/* reset the base class */};
  };

  struct Derived : public Base {
    void reset() {/* reset the derived class, but not the base class */};
  };
