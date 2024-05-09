.. title:: clang-tidy - bugprone-return-const-ref-from-parameter

bugprone-return-const-ref-from-parameter
========================================

Detects return statements that return a constant reference parameter as constant
reference. This may cause use-after-free errors if the caller uses xvalues as
arguments.

In C++, constant reference parameters can accept xvalues which will be destructed
after the call. When the function returns such a parameter also as constant reference,
then the returned reference can be used after the object it refers to has been
destroyed.

Example
-------

.. code-block:: c++

  struct S {
    int v;
    S(int);
    ~S();
  };
  
  const S &fn(const S &a) {
    return a;
  }

  const S& s = fn(S{1});
  s.v; // use after free
