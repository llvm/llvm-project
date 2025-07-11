.. title:: clang-tidy - bugprone-default-lambda-capture

bugprone-default-lambda-capture
===============================

Warns when lambda expressions use default capture modes (``[=]`` or ``[&]``) 
instead of explicitly capturing specific variables. Default captures can make 
code less readable and more error-prone by making it unclear which variables 
are being captured and how.

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
