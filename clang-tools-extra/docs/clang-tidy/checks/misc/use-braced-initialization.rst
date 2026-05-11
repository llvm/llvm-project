.. title:: clang-tidy - misc-use-braced-initialization

misc-use-braced-initialization
==============================

Suggests replacing parenthesized initialization with braced initialization.

Braced initialization has several advantages:

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

  // Member initializer lists:
  struct Widget : Matrix {
    int value;
    Widget() : Matrix(3, 4), value(0) {}
    // -> Widget() : Matrix{3, 4}, value{0} {}
  };

The check skips cases where changing from ``()`` to ``{}`` would alter program
semantics:

- Constructor calls where a ``std::initializer_list`` overload could be
  preferred under braced initialization and all arguments are convertible
  to the list's element type, which would silently change semantics.
  For example, ``std::vector<int> v(5, 1)`` is skipped because
  ``std::vector<int> v{5, 1}`` would create a two-element vector instead
  of five ones, but ``std::string s("hello")`` is diagnosed because
  ``const char*`` cannot implicitly convert to ``char``.
- Direct-initialized ``auto`` variables, where deduction rules may differ
  between C++ standards.
- Expressions in macro expansions.

Limitations
-----------

Braced initialization prohibits implicit narrowing conversions. When
the check detects that changing ``()`` to ``{}`` would introduce a
narrowing conversion, it emits the warning with a note and attaches
the brace fix-it to the note rather than the warning. The fix is not
applied by default, but can be opted into with ``--fix-notes``.

.. code-block:: c++

  struct Foo {
    Foo(short n);
  };

  int n = 10;
  Foo f(n);  // warning + note: narrowing from 'int' to 'short'
             // fix-it on the note, applied only with --fix-notes

References
----------

This check corresponds to the C++ Core Guidelines rule
`ES.23
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#es23-prefer-the--initializer-syntax>`_.
