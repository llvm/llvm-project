.. title:: clang-tidy - misc-use-braced-initialization

misc-use-braced-initialization
==============================

Suggests replacing parenthesized initialization with braced initialization.

Braced initialization has several advantages over parenthesized initialization:

- **Catches narrowing conversions.** ``int x{3.14}`` is a compile-time error,
  while ``int x(3.14)`` silently truncates.
- **Uniform syntax.** Braces work consistently for aggregates, containers, and
  constructors, giving a single initialization style across all types.

For example:

.. code-block:: c++

  struct Matrix {
    Matrix(int rows, int cols);
  };

  // Variable declarations:
  Matrix m(3, 4);          // -> Matrix m{3, 4};
  int n(42);               // -> int n{42};

  // Copy initialization:
  Matrix m = Matrix(3, 4); // -> Matrix m = Matrix{3, 4};

  // Temporary objects:
  use(Matrix(3, 4));       // -> use(Matrix{3, 4});

  // New expressions:
  auto *p = new Matrix(3, 4); // -> auto *p = new Matrix{3, 4};

The check skips cases where changing from ``()`` to ``{}`` would alter program
semantics:

- Types that have any constructor accepting ``std::initializer_list``, since
  braced initialization would prefer that overload and silently change
  semantics.
- Direct-initialized ``auto`` variables, where deduction rules may differ
  between C++ standards.
- Expressions in macro expansions.

.. note::

  Braced initialization prohibits implicit narrowing conversions.
  In some cases the suggested fix may introduce a compiler error.

  .. code-block:: c++

    struct Foo {
      Foo(short n);
    };

    int n = 10;
    Foo f(n);  // OK: implicit narrowing allowed with ()
    Foo f{n};  // error: narrowing conversion from 'int' to 'short'

References
----------

This check corresponds to the C++ Core Guidelines rule
`C++ Core Guidelines ES.23
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#es23-prefer-the--initializer-syntax>`_.
