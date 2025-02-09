.. title:: clang-tidy - modernize-use-using

modernize-use-using
===================

The check converts the usage of ``typedef`` with ``using`` keyword.

Before:

.. code-block:: c++

  typedef int variable;

  class Class{};
  typedef void (Class::* MyPtrType)() const;

  typedef struct { int a; } R_t, *R_p;

After:

.. code-block:: c++

  using variable = int;

  class Class{};
  using MyPtrType = void (Class::*)() const;

  using R_t = struct { int a; };
  using R_p = R_t*;

The checker ignores `typedef` within `extern "C" { ... }` blocks.

.. code-block:: c++

  extern "C" {
    typedef int InExternC; // Left intact.
  }

This check requires using C++11 or higher to run.

Options
-------

.. option:: IgnoreMacros

   If set to `true`, the check will not give warnings inside macros. Default
   is `true`.

.. option:: IgnoreExternC

   If set to `true`, the check will not give warning inside `extern "C"`scope.
   Default is `false`