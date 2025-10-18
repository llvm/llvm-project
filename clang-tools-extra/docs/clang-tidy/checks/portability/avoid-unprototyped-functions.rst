.. title:: clang-tidy - portability-avoid-unprototyped-functions

portability-avoid-unprototyped-functions
========================================

Checks if unprototyped function types are used in the source code.

For example:

.. code-block:: c

  void foo();         // Bad: unprototyped function declaration
  void foo(void);     // Good: a function declaration that takes no parameters

  void (*ptr)();      // Bad: pointer to an unprototyped function
  void (*ptr)(void);  // Good: pointer to a function that takes no parameters

Before C23 ``void foo()`` means a function that takes any number of parameters, so the following snippet is valid.

.. code-block:: c

  // -std=c17
  void foo();

  int main() {
    foo(1, 2, 3);

    return 0;
  }

Starting from C23 however, ``void foo()`` means a function that takes no parameters, so the same snippet is invalid.

.. code-block:: c

  // -std=c23
  void foo();

  int main() {
    foo(1, 2, 3);
  //    ^ error: too many arguments to function call, expected 0, have 3

    return 0;
  }

Similarly a pointer to an unprototyped function binds to any function before C23, so the following snippet is considered valid.

.. code-block:: c

  // -std=c17
  void foo(int x, int y);

  int main() {
    void (*ptr)() = &foo;

    return 0;
  }

From C23 however, the smae pointer will only bind to functions that take no parameters.

.. code-block:: c

  // -std=c23
  void foo(int x, int y);

  int main() {
    void (*ptr)() = &foo;
    //    ^ error: incompatible function pointer types initializing 'void (*)(void)' with an expression of type 'void (*)(int, int)'

    return 0;
  }

Some codebases might explicitly take advantage of the pre-C23 behavior, but these codebases should also be aware that
their solution might not be fully portable between compilers.
