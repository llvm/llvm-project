.. title:: clang-tidy - qt-integer-sign-comparison

qt-integer-sign-comparison
=============================

The qt-integer-sign-comparison check is an alias, please see
:doc:`modernize-use-integer-sign-comparison <../modernize/use-integer-sign-comparison>`
for more information.

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

.. option:: StringsMatchHeader
  A string specifying a include header file to be added by fix-it. Default
  is `<utility>`.
