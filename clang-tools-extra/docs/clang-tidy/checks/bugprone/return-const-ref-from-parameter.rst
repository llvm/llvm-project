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


This issue can be resolved by declaring an overload of the problematic function
where the ``const &`` parameter is instead declared as ``&&``. The developer has
to ensure that the implementation of that function does not produce a
use-after-free, the exact error that this check is warning against.
Marking such an ``&&`` overload as ``deleted``, will silence the warning as 
well. In the case of different ``const &`` parameters being returned depending
on the control flow of the function, an overload where all problematic
``const &`` parameters have been declared as ``&&`` will resolve the issue.

This issue can also be resolved by adding ``[[clang::lifetimebound]]``. Clang
enable ``-Wdangling`` warning by default which can detect mis-uses of the
annotated function. See `lifetimebound attribute <https://clang.llvm.org/docs/AttributeReference.html#id11>`_
for details.

.. code-block:: c++

  const int &f(const int &a [[clang::lifetimebound]]) { return a; } // no warning
  const int &v = f(1); // warning: temporary bound to local reference 'v' will be destroyed at the end of the full-expression [-Wdangling]
