.. title:: clang-tidy - bugprone-parent-virtual-call

bugprone-parent-virtual-call
============================

Detects and fixes calls to grand-...parent virtual methods instead of calls
to overridden parent's virtual methods.

.. code-block:: c++

  struct A {
    int virtual foo() {...}
  };

  struct B: public A {
    int foo() override {...}
  };

  struct C: public B {
    int foo() override { A::foo(); }
  //                     ^^^^^^^^
  // warning: qualified name A::foo refers to a member overridden in subclass; did you mean 'B'?  [bugprone-parent-virtual-call]
  };
