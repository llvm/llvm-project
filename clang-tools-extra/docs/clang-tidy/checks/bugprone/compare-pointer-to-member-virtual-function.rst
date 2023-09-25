.. title:: clang-tidy - bugprone-compare-pointer-to-member-virtual-function

bugprone-compare-pointer-to-member-virtual-function
===================================================

Detects unspecified behavior about equality comparison between pointer to member
virtual function and anything other than null-pointer-constant.

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
      bool r7 = (v2 == &A::f1); // false positive, but potential risk if assigning other value to v2.

      void (A::*v3)(int) = &A::g1;
      bool r8 = (v3 == &A::g1); // ok, no virtual function match void(A::*)(int) signature.
    }

Provide warnings on equality comparisons involve pointers to member virtual
function or variables which is potential pointer to member virtual function and
any entity other than a null-pointer constant.

In certain compilers, virtual function addresses are not conventional pointers
but instead consist of offsets and indexes within a virtual function table
(vtable). Consequently, these pointers may vary between base and derived
classes, leading to unpredictable behavior when compared directly. This issue
becomes particularly challenging when dealing with pointers to pure virtual
functions, as they may not even have a valid address, further complicating
comparisons.

Instead, it is recommended to utilize the ``typeid`` operator or other appropriate
mechanisms for comparing objects to ensure robust and predictable behavior in
your codebase. By heeding this detection and adopting a more reliable comparison
method, you can mitigate potential issues related to unspecified behavior,
especially when dealing with pointers to member virtual functions or pure
virtual functions, thereby improving the overall stability and maintainability
of your code. In scenarios involving pointers to member virtual functions, it's
only advisable to employ ``nullptr`` for comparisons.

Limitations
-----------

Does not analyze values stored in a variable. For variable, only analyze all virtual
methods in the same ``class`` or ``struct`` and diagnose when assigning a pointer
to member virtual function to this variable is possible.
