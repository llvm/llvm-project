.. title:: clang-tidy - modernize-use-static-lambda

modernize-use-static-lambda
===========================

Finds lambdas with an empty capture list (``[]``) that are not already marked
``static`` and suggests adding the ``static`` specifier (introduced in C++23).

A non-capturing lambda does not depend on any enclosing state.  Marking it
``static`` makes that property explicit in the source and allows the compiler
to omit the implicit conversion-to-function-pointer member.

This check requires C++23 or later because ``static`` as a lambda-specifier
was introduced in C++23 (P2564R3).

.. note::
    The ``static`` and ``mutable`` lambda-specifiers are mutually exclusive.
    Mutable lambdas are not diagnosed even when their capture list is empty.

Example
-------

.. code-block:: c++

    auto square = [](int x) { return x * x; };

    auto answer = [] { return 42; };

transforms to:

.. code-block:: c++

    auto square = [](int x) static { return x * x; };

    auto answer = []() static { return 42; };

Note that when the original lambda has no explicit parameter list, ``()`` is
inserted along with ``static``, because a parameter list is required whenever
lambda-specifiers are present.
