.. title:: clang-tidy - bugprone-capturing-this-in-member-variable

bugprone-capturing-this-in-member-variable
==========================================

Finds lambda captures that capture the ``this`` pointer and store it as class
members without handle the copy and move constructors and the assignments.

Capture this in a lambda and store it as a class member is dangerous because
the lambda can outlive the object it captures. Especially when the object is
copied or moved, the captured ``this`` pointer will be implicitly propagated
to the new object. Most of the time, people will believe that the captured
``this`` pointer points to the new object, which will lead to bugs.

.. code-block:: c++

  struct C {
    C() : Captured([this]() -> C const * { return this; }) {}
    std::function<C const *()> Captured;
  };

  void foo() {
    C v1{};
    C v2 = v1; // v2.Captured capture v1's 'this' pointer
    assert(v2.Captured() == v1.Captured()); // v2.Captured capture v1's 'this' pointer
    assert(v2.Captured() == &v2); // assertion failed.
  }

Possible fixes:
  - marking copy and move constructors and assignment operators deleted.
  - using class member method instead of class member variable with function
    object types.
  - passing ``this`` pointer as parameter.

Options
-------

.. option:: FunctionWrapperTypes

  A semicolon-separated list of names of types. Used to specify function
  wrapper that can hold lambda expressions.
  Default is `::std::function;::std::move_only_function;::boost::function`.

.. option:: BindFunctions

  A semicolon-separated list of fully qualified names of functions that can
  capture ``this`` pointer.
  Default is `::std::bind;::boost::bind;::std::bind_front;::std::bind_back;
  ::boost::compat::bind_front;::boost::compat::bind_back`.
