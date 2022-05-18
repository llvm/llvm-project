.. title:: clang-tidy - readability-non-const-parameter

readability-non-const-parameter
===============================

The check finds function parameters of a pointer type that could be changed to
point to a constant type instead.

When ``const`` is used properly, many mistakes can be avoided. Advantages when
using ``const`` properly:

- prevent unintentional modification of data;

- get additional warnings such as using uninitialized data;

- make it easier for developers to see possible side effects.

This check is not strict about constness, it only warns when the constness will
make the function interface safer.

.. code-block:: c++

  // warning here; the declaration "const char *p" would make the function
  // interface safer.
  char f1(char *p) {
    return *p;
  }

  // no warning; the declaration could be more const "const int * const p" but
  // that does not make the function interface safer.
  int f2(const int *p) {
    return *p;
  }

  // no warning; making x const does not make the function interface safer
  int f3(int x) {
    return x;
  }

  // no warning; Technically, *p can be const ("const struct S *p"). But making
  // *p const could be misleading. People might think that it's safe to pass
  // const data to this function.
  struct S { int *a; int *b; };
  int f3(struct S *p) {
    *(p->a) = 0;
  }

  // no warning; p is referenced by an lvalue.
  void f4(int *p) {
    int &x = *p;
  }
