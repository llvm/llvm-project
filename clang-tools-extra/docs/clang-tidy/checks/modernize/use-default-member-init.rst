.. title:: clang-tidy - modernize-use-default-member-init

modernize-use-default-member-init
=================================

This check converts constructors' member initializers into the new
default member initializers in C++11. Other member initializers that match the
default member initializer are removed. This can reduce repeated code or allow
use of '= default'.

.. code-block:: c++

  struct A {
    A() : i(5), j(10.0) {}
    A(int i) : i(i), j(10.0) {}
    int i;
    double j;
  };

  // becomes

  struct A {
    A() {}
    A(int i) : i(i) {}
    int i{5};
    double j{10.0};
  };

.. note::
  Only converts member initializers for built-in types, enums, and pointers.
  The `readability-redundant-member-init` check will remove redundant member
  initializers for classes.

Options
-------

.. option:: UseAssignment

   If this option is set to `true` (default is `false`), the check will initialize
   members with an assignment. For example:

.. code-block:: c++

  struct A {
    A() {}
    A(int i) : i(i) {}
    int i = 5;
    double j = 10.0;
  };

.. option:: IgnoreMacros

   If this option is set to `true` (default is `true`), the check will not warn
   about members declared inside macros.
