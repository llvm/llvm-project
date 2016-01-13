misc-virtual-near-miss
======================

Warn if a function is a near miss (ie. the name is very similar and the function signiture is the same) to a virtual function from a base class.

Example:

.. code-block:: c++

  struct Base {
    virtual void func();
  };

  struct Derived : Base {
    virtual funk();
    // warning: Do you want to override 'func'?
  };
