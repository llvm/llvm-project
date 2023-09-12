.. title:: clang-tidy - bugprone-compare-pointer-to-member-virtual-function

bugprone-compare-pointer-to-member-virtual-function
===================================================

Detects unspecified behavior about equality comparison between pointer to member virtual 
function and anything other than null-pointer-constant.


.. code-block:: c++
    struct A {
      void f1();
      void f2();
      virtual void f3();
      virtual void f4();

      void g1(int);
    };

    void fn() {
      bool r1 = (&A::f1 == &A::f2);  // ok
      bool r2 = (&A::f1 == &A::f3);  // bugprone
      bool r3 = (&A::f1 != &A::f3);  // bugprone
      bool r4 = (&A::f3 == nullptr); // ok
      bool r5 = (&A::f3 == &A::f4);  // bugprone

      void (A::*v1)() = &A::f3;
      bool r6 = (v1 == &A::f1); // bugprone
      bool r6 = (v1 == nullptr); // ok

      void (A::*v2)() = &A::f2;
      bool r7 = (v2 == &A::f1); // false positive

      void (A::*v3)(int) = &A::g1;
      bool r8 = (v3 == &A::g1); // ok, no virtual function match void(A::*)(int) signature
    }


Limitations
-----------

The check will not analyze values stored in a variable. For variable, the check will analyze all
virtual methods in the same ``class`` or ``struct`` and diagnose when assigning a pointer to member
virtual function to this variable is possible.
