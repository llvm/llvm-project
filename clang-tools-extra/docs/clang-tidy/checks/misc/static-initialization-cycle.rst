.. title:: clang-tidy - misc-static-initialization-cycle

misc-static-initialization-cycle
================================

Finds cyclical initialization of static variables.

The cycle can come from reference to static variables or from (static) function
calls during initialization. Such cycles can cause undefined behavior. In this
context "static" means C++ ``static`` class members, global variables, global
functions, and ``static`` variables inside functions.

For the purpose of this check, the initialization of a static variable
*uses* another static variable or function if it appears in the initializer
expression. A function *uses* a static variable or function if the variable
or function appears at any place in the function code (except if the variable
is assigned to). The check can detect cycles in this "usage graph".

The check does not consider conditions in function code and does not follow the
value of static variables (if assigned to another variable). For this reason it
can produce false positives in some cases.

Examples
--------

.. code-block:: c++

  struct S { static int A; };
  int B = S::A;
  int S::A = B;

Cycle in variable initialization.

.. code-block:: c++

  int f1(int X, int Y);

  struct S { static int A; };

  int B = S::A + 1;
  int S::A = f1(B, 2);

Cyclical initialization: ``B`` uses value of ``S::A``, and ``S::A`` may use
value of ``B`` (the check gives always warning regardless of the code of
``f1``).

.. code-block:: c++

  struct S { static int A; };
  int f1() {
    return S::A;
  }
  int S::A = f1();

This code results in initialization of ``S::A`` with itself through a function
call. The check would emit a warning in any case when ``S::A`` appears in
``f1`` (even if the return value is not affected by it).

References
----------

* CERT C++ Coding Standard rule `DCL56-CPP. Avoid cycles during initialization
  of static objects <https://wiki.sei.cmu.edu/confluence/display/cplusplus/DCL56-CPP.+Avoid+cycles+during+initialization+of+static+objects>`_.
