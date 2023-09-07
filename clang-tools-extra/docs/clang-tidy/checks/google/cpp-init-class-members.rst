.. title:: clang-tidy - google-cpp-init-class-members

google-cpp-init-class-members
=============================

Checks that class members are initialized in constructors (implicitly or
explicitly). Reports constructors or classes where class members are not
initialized. The goal of this check is to eliminate UUM (Use of
Uninitialized Memory) bugs caused by uninitialized class members.

This check is under active development: the check authors made a few commits
and are actively working on more commits. Users who want a mature and stable
check should not use this check yet.

This check is different from ProTypeMemberInitCheck in that this check
attempts to eliminate UUMs as a bug class, at the expense of false
positives. The authors of this check will add more documentation about the
differences with ProTypeMemberInitCheck as the check evolves.

For now, this check reports `X` in the following two patterns:

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
