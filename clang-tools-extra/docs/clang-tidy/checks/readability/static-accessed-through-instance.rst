.. title:: clang-tidy - readability-static-accessed-through-instance

readability-static-accessed-through-instance
============================================

Checks for member expressions that access static members through instances, and
replaces them with uses of the appropriate qualified-id.

Example:

The following code:

.. code-block:: c++

  struct C {
    static void foo();
    static int x;
    enum { E1 };
    enum E { E2 };
  };

  C *c1 = new C();
  c1->foo();
  c1->x;
  c1->E1;
  c1->E2;

is changed to:

.. code-block:: c++

  C *c1 = new C();
  C::foo();
  C::x;
  C::E1;
  C::E2;

