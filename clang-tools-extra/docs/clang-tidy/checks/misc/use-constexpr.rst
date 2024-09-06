.. title:: clang-tidy - misc-use-constexpr

misc-use-constexpr
==================

Find functions and variables that can be declared 'constexpr'.

The check analyses any function and variable according to the rules defined
for the language version that the code is compiled with.
Changing to a newer language standard may therefore offer additional opportunity
to declare a function or variable as ``constexpr``.

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

  This type is a literal type, but can not be constructed at compile-time,
  so with `ConservativeLiteralType` equal to `true`, variables or funtions
  with this type are not considered to support ``constexpr``. Default is
  `true`.

.. option:: AddConstexprToMethodOfClassWithoutConstexprConstructor

  While a function of a class or struct could be declared ``constexpr``, when
  the class itself can never be constructed at compile-time, then adding
  ``constexpr`` to a member function is superfluous. This option controls if
  ``constexpr`` should be added anyways. Default is ``false``.

