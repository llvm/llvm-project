.. title:: clang-tidy - modernize-use-va-opt

modernize-use-va-opt
====================

Suggest using ``__VA_OPT__(,)`` instead of ``, ##__VA_ARGS__`` when implementing
variadic macro. ``, ##__VA_ARGS__`` is a GNU extension.

.. code:: c++

    extern int bar(...);
    #define FOO(a, ...) bar(a, ##__VA_ARGS__)

becomes:

.. code:: c++

    extern int bar(...);
    #define FOO(a, ...) bar(a __VA_OPT__(,) __VA_ARGS__)
