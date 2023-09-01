.. title:: clang-tidy - google-cpp-init-class-members

google-cpp-init-class-members
=============================

Checks that class members are initialized in constructors (implicitly or
explicitly). Reports constructors or classes where class members are not
initialized. The goal of this checker is to eliminate UUM (Use of
Uninitialized Memory) bugs caused by uninitialized class members.

This checker is different from ProTypeMemberInitCheck in that this checker
attempts to eliminate UUMs as a bug class, at the expense of false
positives.

This checker is WIP. We are incrementally adding features and increasing
coverage until we get to a shape that is acceptable.

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
