.. title:: clang-tidy - bugprone-virtual-near-miss

bugprone-virtual-near-miss
==========================

Warn if a function is a near miss (i.e. the name is very similar and the function
signature is the same) to a virtual function from a base class.

Example:

.. code-block:: c++

  struct Base {
    virtual void func();
  };

  struct Derived : Base {
    virtual void funk();
    // warning: 'Derived::funk' has a similar name and the same signature as virtual method 'Base::func'; did you mean to override it?
  };
