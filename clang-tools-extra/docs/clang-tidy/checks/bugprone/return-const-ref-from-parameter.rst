.. title:: clang-tidy - bugprone-return-const-ref-from-parameter

bugprone-return-const-ref-from-parameter
========================================

Detects the function which returns the const reference from parameter which
causes potential use after free if the caller uses xvalue as argument.

In c++, const reference parameter can accept xvalue which will be destructed
after the call. When the function returns this parameter also as const reference,
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
