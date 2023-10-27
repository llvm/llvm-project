.. title:: clang-tidy - bugprone-multiple-new-in-one-expression

bugprone-multiple-new-in-one-expression
=======================================

Finds multiple ``new`` operator calls in a single expression, where the
allocated memory by the first ``new`` may leak if the second allocation fails
and throws exception.

C++ does often not specify the exact order of evaluation of the operands of an
operator or arguments of a function. Therefore if a first allocation succeeds
and a second fails, in an exception handler it is not possible to tell which
allocation has failed and free the memory. Even if the order is fixed the result
of a first ``new`` may be stored in a temporary location that is not reachable
at the time when a second allocation fails. It is best to avoid any expression
that contains more than one ``operator new`` call, if exception handling is
used to check for allocation errors.

Different rules apply for are the short-circuit operators ``||`` and ``&&`` and
the ``,`` operator, where evaluation of one side must be completed before the
other starts. Expressions of a list-initialization (initialization or
construction using ``{`` and ``}`` characters) are evaluated in fixed order.
Similarly, condition of a ``?`` operator is evaluated before the branches are
evaluated.

The check reports warning if two ``new`` calls appear in one expression at
different sides of an operator, or if ``new`` calls appear in different
arguments of a function call (that can be an object construction with ``()``
syntax). These ``new`` calls can be nested at any level.
For any warning to be emitted the ``new`` calls should be in a code block where
exception handling is used with catch for ``std::bad_alloc`` or
``std::exception``. At ``||``, ``&&``, ``,``, ``?`` (condition and one branch)
operators no warning is emitted. No warning is emitted if both of the memory
allocations are not assigned to a variable or not passed directly to a function.
The reason is that in this case the memory may be intentionally not freed or the
allocated objects can be self-destructing objects.

Examples:

.. code-block:: c++

  struct A {
    int Var;
  };
  struct B {
    B();
    B(A *);
    int Var;
  };
  struct C {
    int *X1;
    int *X2;
  };

  void f(A *, B *);
  int f1(A *);
  int f1(B *);
  bool f2(A *);

  void foo() {
    A *PtrA;
    B *PtrB;
    try {
      // Allocation of 'B'/'A' may fail after memory for 'A'/'B' was allocated.
      f(new A, new B); // warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception; order of these allocations is undefined

      // List (aggregate) initialization is used.
      C C1{new int, new int}; // no warning

      // Allocation of 'B'/'A' may fail after memory for 'A'/'B' was allocated but not yet passed to function 'f1'.
      int X = f1(new A) + f1(new B); // warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception; order of these allocations is undefined

      // Allocation of 'B' may fail after memory for 'A' was allocated.
      // From C++17 on memory for 'B' is allocated first but still may leak if allocation of 'A' fails.
      PtrB = new B(new A); // warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception

      // 'new A' and 'new B' may be performed in any order.
      // 'new B'/'new A' may fail after memory for 'A'/'B' was allocated but not assigned to 'PtrA'/'PtrB'.
      (PtrA = new A)->Var = (PtrB = new B)->Var; // warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception; order of these allocations is undefined

      // Evaluation of 'f2(new A)' must be finished before 'f1(new B)' starts.
      // If 'new B' fails the allocated memory for 'A' is supposedly handled correctly because function 'f2' could take the ownership.
      bool Z = f2(new A) || f1(new B); // no warning

      X = (f2(new A) ? f1(new A) : f1(new B)); // no warning

      // No warning if the result of both allocations is not passed to a function
      // or stored in a variable.
      (new A)->Var = (new B)->Var; // no warning

      // No warning if at least one non-throwing allocation is used.
      f(new(std::nothrow) A, new B); // no warning
    } catch(std::bad_alloc) {
    }

    // No warning if the allocation is outside a try block (or no catch handler exists for std::bad_alloc).
    // (The fact if exceptions can escape from 'foo' is not taken into account.)
    f(new A, new B); // no warning
  }
