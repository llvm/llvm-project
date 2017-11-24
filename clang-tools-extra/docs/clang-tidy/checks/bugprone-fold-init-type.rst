.. title:: clang-tidy - bugprone-fold-init-type

bugprone-fold-init-type
=======================

The check flags type mismatches in
`folds <https://en.wikipedia.org/wiki/Fold_(higher-order_function)>`_
like ``std::accumulate`` that might result in loss of precision.
``std::accumulate`` folds an input range into an initial value using the type of
the latter, with ``operator+`` by default. This can cause loss of precision
through:

- Truncation: The following code uses a floating point range and an int
  initial value, so trucation wil happen at every application of ``operator+``
  and the result will be `0`, which might not be what the user expected.

.. code-block:: c++

  auto a = {0.5f, 0.5f, 0.5f, 0.5f};
  return std::accumulate(std::begin(a), std::end(a), 0);

- Overflow: The following code also returns `0`.

.. code-block:: c++

  auto a = {65536LL * 65536 * 65536};
  return std::accumulate(std::begin(a), std::end(a), 0);
