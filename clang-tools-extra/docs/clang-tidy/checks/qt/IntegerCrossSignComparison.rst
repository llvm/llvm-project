.. title:: clang-tidy - qt-integer-sign-comparison

qt-integer-sign-comparison
=============================

The check detects comparison between signed and unsigned integer values.
If C++20 is supported, the check suggests a std related fix-it.
If C++17 is supported, the check suggests a Qt related fix-it.

Examples of fixes created by the check:

.. code-block:: c++

    uint func(int bla)
    {
        uint result;
        if (result == bla)
            return 0;

        return 1;
    }

in C++17 becomes

.. code-block:: c++

    <QtCore/q20utility.h>

    uint func(int bla)
    {
        uint result;
        if (q20::cmp_equal(result, bla))
            return 0;

        return 1;
    }

in C++20 becomes

.. code-block:: c++

    #include <utility>

    uint func(int bla)
    {
        uint result;
        if (std::cmp_equal(result, bla))
            return 0;

        return 1;
    }

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.

.. option:: IsQtApplication

  When `true` (default `false`), then it is assumed that the code being analyzed
  is the Qt-based code.
