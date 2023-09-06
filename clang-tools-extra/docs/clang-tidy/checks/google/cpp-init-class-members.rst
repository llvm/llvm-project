.. title:: clang-tidy - google-cpp-init-class-members

google-cpp-init-class-members
=============================

Checks that class members are initialized in constructors (implicitly or
explicitly). Reports constructors or classes where class members are not
initialized. The goal of this checker is to eliminate UUM (Use of
Uninitialized Memory) bugs caused by uninitialized class members.

This checker is under active development: the checker authors made a few commits
and are actively working on more commits. Users who want a mature and stable
checker should not use this checker yet.

This checker is different from ProTypeMemberInitCheck in that this checker
attempts to eliminate UUMs as a bug class, at the expense of false
positives. The authors of this checker will add more documentation about the
differences with ProTypeMemberInitCheck as the checker evolves.

For now, this checker reports `X` in the following two patterns:

.. code-block:: c++
  class SomeClass {
  public:
    SomeClass() = default;

  private:
    int X;
  };

.. code-block:: c++
  struct SomeStruct {
    int X;
  };
