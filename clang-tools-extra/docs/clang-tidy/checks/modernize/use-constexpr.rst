.. title:: clang-tidy - modernize-use-constexpr

modernize-use-constexpr
=======================

Finds functions and variables that can be declared ``constexpr``.

This check currently supports the ``constexpr`` rule-set of C++11.

The check analyses any function and variable according to the rules defined
for the language version that the code compiles with.
Changing to a newer language standard may therefore offer additional
opportunities to declare a function or variable as ``constexpr``.
Furthermore, this check can be incremental in terms of its diagnostics. For
example, declaring a function ``constepxr`` might create new opportunities of
marking additional variables or function ``constexpr``, which can only be found
in subsequent runs of this check.

Before C++23, ``static constexpr`` variables could not be declared inside a
``constexpr`` function. This check prefers adding ``constexpr`` to an enclosing
function over adding ``constexpr`` to a static local variable inside that
function.

Limitations
-----------

* Only analyzes variables declared ``const``, because this check would have
  to duplicate the expensive analysis of the 
  :doc:`misc-const-correctness<../misc/const-correctness>` check.
  For the best results, enable both `misc-const-correctness` and
  `modernize-use-constexpr` together.

* Only analyzes variable declarations that declare a single variable

Options
-------

.. option:: ConservativeLiteralType

  With this option enabled, only literal types that can be constructed at
  compile-time are considered to support ``constexpr``.

  .. code-block:: c++

    struct NonLiteral{
      NonLiteral();
      ~NonLiteral();
      int &ref;
    };

  This type is a literal type, but can not be constructed at compile-time.
  With :option:`ConservativeLiteralType` equal to `true`, variables or functions
  with this type are not diagnosed to add ``constexpr``. Default is
  `true`.

.. option:: AddConstexprToMethodOfClassWithoutConstexprConstructor

  While a function of a class or struct could be declared ``constexpr``, when
  the class itself can never be constructed at compile-time, then adding
  ``constexpr`` to a member function is superfluous. This option controls if
  ``constexpr`` should be added anyways. Default is `false`.

.. option:: ConstexprString

  The string to use to specify a variable or function as ``constexpr``, for
  example, a macro. Default is `constexpr`.

.. option:: ConstexprString

  The string to use with C++23 to specify a function-local variable as 
  ``static constexpr``, for example, a macro. Default is `static constexpr`
  (concatenating `static` with the :option:`ConstexprString`).
