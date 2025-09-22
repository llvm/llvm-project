.. title:: clang-tidy - bugprone-default-lambda-capture

bugprone-default-lambda-capture
===============================

  Finds lambda expressions that use default capture modes (``[=]`` or ``[&]``)
  and suggests using explicit captures instead. Default captures can lead to
  subtle bugs including dangling references with ``[&]``, unnecessary copies
  with ``[=]``, and make code less maintainable by hiding which variables are
  actually being captured.

Implements Item 31 of Effective Modern C++ by Scott Meyers.

Example
-------

.. code-block:: c++

  void example() {
    int x = 1;
    int y = 2;
    
    // Bad - default capture by copy
    auto lambda1 = [=]() { return x + y; };
    
    // Bad - default capture by reference
    auto lambda2 = [&]() { return x + y; };
    
    // Good - explicit captures
    auto lambda3 = [x, y]() { return x + y; };
    auto lambda4 = [&x, &y]() { return x + y; };
  }

The check will warn on:

- Default capture by copy: ``[=]``
- Default capture by reference: ``[&]``
- Mixed captures with defaults: ``[=, &x]`` or ``[&, x]``

The check will not warn on:

- Explicit captures: ``[x]``, ``[&x]``, ``[x, y]``, ``[this]``
- Empty capture lists: ``[]``
