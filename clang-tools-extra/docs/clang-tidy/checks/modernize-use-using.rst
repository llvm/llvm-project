.. title:: clang-tidy - modernize-use-using

modernize-use-using
===================

Use C++11's ``using`` instead of ``typedef``.

Before:

.. code:: c++

  typedef int variable;

  class Class{};
  typedef void (Class::* MyPtrType)() const;

After:

.. code:: c++

  using variable = int;

  class Class{};
  using MyPtrType = void (Class::*)() const;

This check requires using C++11 or higher to run.
