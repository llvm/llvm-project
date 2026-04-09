.. title:: clang-tidy - modernize-use-std-tie

modernize-use-std-tie
=====================

Suggests replacing manual field-by-field lexicographical comparisons with ``std::tie``.

Manual implementations of ``operator<`` and ``operator>`` can be error-prone and verbose.
Using ``std::tie`` makes the code more readable and optimizes branch prediction.

Example:

.. code-block:: c++

  bool operator<(const A& lhs, const A& rhs) {
    if (lhs.n != rhs.n) {
      return lhs.n < rhs.n;
    }
    return lhs.s < rhs.s;
  }

Transforms to:

.. code-block:: c++

  bool operator<(const A& lhs, const A& rhs) {
    return std::tie(lhs.n, lhs.s) < std::tie(rhs.n, rhs.s);
  }