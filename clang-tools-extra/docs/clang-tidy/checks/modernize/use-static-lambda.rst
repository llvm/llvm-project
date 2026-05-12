.. title:: clang-tidy - modernize-use-static-lambda

modernize-use-static-lambda
===========================

Finds lambdas with an empty capture list (``[]``) that are not already marked
``static`` and suggests adding the ``static`` specifier (introduced in C++23).

Marking a non-capturing lambda ``static`` turns ``operator()`` into a static
member function, making it clear that the lambda has no dependency on any
closure state.

Example
-------

.. code-block:: c++

    auto square = [](int x) { return x * x; };

    auto answer = [] { return 42; };

transforms to:

.. code-block:: c++

    auto square = [](int x) static { return x * x; };

    auto answer = [] static { return 42; };

