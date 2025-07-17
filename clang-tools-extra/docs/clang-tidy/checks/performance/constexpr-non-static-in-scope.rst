.. title:: clang-tidy - performance-constexpr-non-static-in-scope

performance-constexpr-non-static-in-scope
=========================================

The check ``performance-constexpr-non-static-in-scope`` identifies ``constexpr`` variables declared in function (local) scope that are not marked ``static``. In most cases, such variables should be declared `static constexpr` to avoid creating a new instance on every function call.

The check will always warn for non-static ``constexpr`` variables in non-constexpr functions. For ``constexpr`` functions, the check warns only in C++23 and newer (and this behavior can be controlled with the ``WarnInConstexprFuncCpp23`` option).

For example:
When a ``constexpr` ` is declared without ``static``:
.. code-block:: c++
    // BEFORE
    void foo() {
        constexpr int x = 42;
    }

    // AFTER
    void foo() {
        static constexpr int x = 42; // Corrected to static constexpr
    }

When a ``constexpr`` is declared in a ``constexpr`` function without ``static`` (only in C++23 and newer):
.. code-block:: c++
    // BEFORE
    constexpr int bar() {
        constexpr int y = 123;
        return y;
    }

    // AFTER
    constexpr int bar() {
        static constexpr int y = 123; // Corrected to static constexpr
        return y;
    }

Options
-------

.. option:: WarnInConstexprFuncCpp23

   When true (default), warns on ``constexpr`` variables inside ``constexpr`` functions in C++23 and newer. Set to ``false`` to disable this warning in that scenario.
