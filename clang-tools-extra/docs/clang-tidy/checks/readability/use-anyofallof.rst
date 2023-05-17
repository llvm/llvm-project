.. title:: clang-tidy - readability-use-anyofallof

readability-use-anyofallof
==========================

Finds range-based for loops that can be replaced by a call to ``std::any_of`` or
``std::all_of``. In C++ 20 mode, suggests ``std::ranges::any_of`` or
``std::ranges::all_of``.

Example:

.. code-block:: c++

  bool all_even(std::vector<int> V) {
    for (int I : V) {
      if (I % 2)
        return false;
    }
    return true;
    // Replace loop by
    // return std::ranges::all_of(V, [](int I) { return I % 2 == 0; });
  }
