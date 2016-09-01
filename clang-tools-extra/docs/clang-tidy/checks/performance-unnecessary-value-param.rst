.. title:: clang-tidy - performance-unnecessary-value-param

performance-unnecessary-value-param
===================================

Flags value parameter declarations of expensive to copy types that are copied
for each invocation but it would suffice to pass them by const reference.

The check is only applied to parameters of types that are expensive to copy
which means they are not trivially copyable or have a non-trivial copy
constructor or destructor.

To ensure that it is safe to replace the value parameter with a const reference
the following heuristic is employed:

1. the parameter is const qualified;
2. the parameter is not const, but only const methods or operators are invoked
   on it, or it is used as const reference or value argument in constructors or
   function calls.

Example:

.. code-block:: c++

  void f(const string Value) {
    // The warning will suggest making Value a reference.
  }

  void g(ExpensiveToCopy Value) {
    // The warning will suggest making Value a const reference.
    Value.ConstMethd();
    ExpensiveToCopy Copy(Value);
  }

If the parameter is not const, only copied or assigned once and has a
non-trivial move-constructor or move-assignment operator respectively the check
will suggest to move it.

Example:

.. code-block:: c++

  void setValue(string Value) {
    Field = Value;
  }

Will become:

.. code-block:: c++

  #include <utility>

  void setValue(string Value) {
    Field = std::move(Value);
  }

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.
