.. title:: clang-tidy - modernize-use-integer-sign-comparison

modernize-use-integer-sign-comparison
=====================================

Replace comparisons between signed and unsigned integers with their safe
C++20 ``std::cmp_*`` alternative, if available.

The check provides a replacement only for C++20 or later, otherwise
it highlights the problem and expects the user to fix it manually.

Examples of fixes created by the check:

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

Options
-------

.. option:: IncludeStyle

  A string specifying which include-style is used, `llvm` or `google`.
  Default is `llvm`.

.. option:: EnableQtSupport

  Makes C++17 ``q20::cmp_*`` alternative available for Qt-based
  applications. Default is `false`.
