.. title:: clang-tidy - modernize-use-static-lambda

modernize-use-static-lambda
===========================

Finds lambdas with an empty capture list (``[]``) that are not already marked
``static`` and suggests adding the ``static`` specifier (introduced in C++23).

A non-capturing lambda does not depend on any enclosing state.  Marking it
``static`` makes that property explicit in the source and enables the implicit
conversion-to-function-pointer to return a direct pointer to ``operator()``
rather than a trampoline wrapping a default-constructed closure object.

Example
-------

.. code-block:: c++

    auto square = [](int x) { return x * x; };

    auto answer = [] { return 42; };

transforms to:

.. code-block:: c++

    auto square = [](int x) static { return x * x; };

    auto answer = [] static { return 42; };

When the original lambda has no explicit parameter list, ``static`` is inserted
directly after the capture list (``[]``) without adding ``()``.  This is valid
in C++23, where lambda-specifiers may appear without an explicit parameter list
(grammar form 4: ``[captures] specs { body }``).
