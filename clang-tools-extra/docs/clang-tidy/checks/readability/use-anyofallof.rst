.. title:: clang-tidy - readability-use-anyofallof

readability-use-anyofallof
==========================

Finds range-based for loops that can be replaced by a call to
``std::any_of`` or ``std::all_of``. In C++20 mode, suggests
``std::ranges::any_of`` or ``std::ranges::all_of``.

Example:

.. code-block:: c++

  bool all_even(const std::vector<int> &V) {
    for (int I : V) {
      if (I % 2)
        return false;
    }
    return true;
  }
  // Replace loop by
  // return std::ranges::all_of(V, [](int I) { return I % 2 == 0; }); (C++20)
  // return std::all_of(V.begin(), V.end(), [](int I) { return I % 2 == 0; }); (pre-C++20)

When using a raw initializer list or a temporary range (pre-C++20), it's
recommended to materialize it in a local variable first to avoid potential
lifetime issues. In C++20, temporary ranges can be used directly with
``std::ranges`` algorithms as they handle the lifetime of a temporary range
correctly.

Example with raw initializer list:

.. code-block:: c++

  bool contains_zero(int a, int b, int c) {
    for (int i : {a, b, c}) {
      if (i == 0)
        return true;
    }
    return false;
  }
  // Replace loop by
  // auto range = {a, b, c};
  // return std::ranges::any_of(range, [](int i) { return i == 0; }); (C++20)
  // return std::any_of(range.begin(), range.end(), [](int i) { return i == 0; }); (pre-C++20)

Example with temporary range:

.. code-block:: c++

  extern std::vector<int> get_values();

  bool has_even() {
    for (int i : get_values()) {
      if (i % 2 == 0)
        return true;
    }
    return false;
  }
  // Replace loop by
  // return std::ranges::any_of(get_values(), [](int i) { return i % 2 == 0; }); (C++20)
  //
  // auto values = get_values();
  // return std::any_of(values.begin(), values.end(), [](int i) { return i % 2 == 0; }); (pre-C++20)
