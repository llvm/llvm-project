.. title:: clang-tidy - modernize-use-integer-sign-comparison

modernize-use-integer-sign-comparison
=====================================

Performs comparisons between signed and unsigned integer types
mathematically correct. If C++20 is supported a fix-it replaces
integers comparisons to ``std::cmp_equal``, ``std::cmp_not_equal``,
``std::cmp_less``, ``std::cmp_greater``, ``std::cmp_less_equal`` and
``std::cmp_greater_equal`` functions.

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
