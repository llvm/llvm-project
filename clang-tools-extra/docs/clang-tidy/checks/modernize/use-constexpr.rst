.. title:: clang-tidy - modernize-use-constexpr

modernize-use-constexpr
=======================

Finds functions and variables that can be declared ``constexpr``.

The check analyses any function and variable according to the rules defined
for the language version that the code is compiled with.
Changing to a newer language standard may therefore offer additional opportunity
to declare a function or variable as ``constexpr``.
Furthermore, this check can be incremental in terms of its diagnostics. For
example, declaring a function ``constepxr`` might create new opportunities of
marking additional variables or function ``constexpr``, which can only be found
in subsequent runs of this check.

For variables, the check will only detect variables that can be declared
``constexpr`` if they are already ``const``.
This is because this check would have to duplicate the expensive analysis of the
:doc:`misc-const-correctness<../misc/const-correctness>` check.
Therefore, it is recommended to have 
:doc:`misc-const-correctness<../misc/const-correctness>` enabled
in the Clang-Tidy config when this check is, so that all opportunities for
``const`` and also ``constexpr`` are explored.

Options
-------

.. option:: ConservativeLiteralType

  With this option enabled, only literal types that can be constructed at
  compile-time are considered to supoprt ``constexpr``.

  .. code-block:: c++

    struct NonLiteral{
      NonLiteral();
      ~NonLiteral();
      int &ref;
    };

  This type is a literal type, but can not be constructed at compile-time.
  With `ConservativeLiteralType` equal to `true`, variables or funtions
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
  (concatenating `static` with the `ConstexprString` option).

