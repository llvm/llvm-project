.. title:: clang-tidy - modernize-use-integer-sign-comparison

modernize-use-integer-sign-comparison
=====================================

Modernizes integer comparisons using C++20 (or Qt's C++17 backport) safe
comparison utilities from ``<utility>``:

- Replaces comparisons between signed and unsigned integers with
  ``std::cmp_*`` alternatives.
- Replaces manual integer range checks using ``std::numeric_limits`` with
  ``std::in_range``.

Both transformations correctly handle signed/unsigned boundaries without
the implicit conversion pitfalls of the original code.

**Sign comparison** — replaces mixed-sign comparisons:

.. code-block:: c++

  unsigned int func(int a, unsigned int b) {
    return a == b;
  }

becomes

.. code-block:: c++

  #include <utility>

  unsigned int func(int a, unsigned int b) {
    return std::cmp_equal(a, b);
  }

The check provides a replacement only for C++20 or later, otherwise
it highlights the problem and expects the user to fix it manually.

**Range check** — replaces manual ``numeric_limits`` range checks:

.. code-block:: c++

  #include <limits>
  bool fits_in_int(long val) {
    return val >= std::numeric_limits<int>::min() &&
           val <= std::numeric_limits<int>::max();
  }

becomes

.. code-block:: c++

  #include <limits>
  #include <utility>
  bool fits_in_int(long val) {
    return std::in_range<int>(val);
  }

The check also recognizes negated forms such as
``!(val < std::numeric_limits<T>::min() || val > std::numeric_limits<T>::max())``,
commutative variants of the comparison operators, and ``lowest()`` as
equivalent to ``min()`` for integer types. Typedef aliases are preserved
in the fix-it (e.g. ``int32_t`` rather than ``int``).

Options
-------

.. option:: IncludeStyle

  A string specifying which include-style is used, `llvm` or `google`.
  Default is `llvm`.

.. option:: EnableQtSupport

  Makes C++17 ``q20::cmp_*`` and ``q20::in_range`` alternatives available
  for Qt-based applications. Default is `false`.
