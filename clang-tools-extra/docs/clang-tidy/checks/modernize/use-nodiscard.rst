.. title:: clang-tidy - modernize-use-nodiscard

modernize-use-nodiscard
=======================

Adds ``[[nodiscard]]`` attributes (introduced in C++17) to member functions in
order to highlight at compile time which return values should not be ignored.

Member functions need to satisfy the following conditions to be considered by
this check:

 - no ``[[nodiscard]]``, ``[[noreturn]]``,
   ``__attribute__((warn_unused_result))``,
   ``[[clang::warn_unused_result]]`` nor ``[[gcc::warn_unused_result]]``
   attribute,
 - non-void return type,
 - non-template return types,
 - const member function,
 - non-variadic functions,
 - no non-const reference parameters,
 - no pointer parameters,
 - no template parameters,
 - no template function parameters,
 - not be a member of a class with mutable member variables,
 - no Lambdas,
 - no conversion functions.

Such functions have no means of altering any state or passing values other than
via the return type. Unless the member functions are altering state via some
external call (e.g. I/O).

Example
-------

.. code-block:: c++

    bool empty() const;
    bool empty(int i) const;

transforms to:

.. code-block:: c++

    [[nodiscard]] bool empty() const;
    [[nodiscard]] bool empty(int i) const;

Options
-------

.. option:: ReplacementString

    Specifies a macro to use instead of ``[[nodiscard]]``. This is useful when
    maintaining source code that needs to compile with a pre-C++17 compiler.

Example
^^^^^^^

.. code-block:: c++

    bool empty() const;
    bool empty(int i) const;

transforms to:

.. code-block:: c++

    NO_DISCARD bool empty() const;
    NO_DISCARD bool empty(int i) const;

if the :option:`ReplacementString` option is set to `NO_DISCARD`.

.. note::

    If the :option:`ReplacementString` is not a C++ attribute, but instead a
    macro, then that macro must be defined in scope or the fix-it will not be
    applied.

.. note::

    For alternative ``__attribute__`` syntax options to mark functions as
    ``[[nodiscard]]`` in non-c++17 source code.
    See https://clang.llvm.org/docs/AttributeReference.html#nodiscard-warn-unused-result
