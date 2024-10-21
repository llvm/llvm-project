.. title:: clang-tidy - modernize-use-integer-sign-comparison

modernize-use-integer-sign-comparison
=====================================

The check detects comparison between signed and unsigned integer values.
If C++20 is supported, the check suggests a fix-it.

Examples of fixes created by the check:

.. code-block:: c++

    uint func(int bla)
    {
        uint result;
        if (result == bla)
            return 0;

        return 1;
    }

becomes

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
