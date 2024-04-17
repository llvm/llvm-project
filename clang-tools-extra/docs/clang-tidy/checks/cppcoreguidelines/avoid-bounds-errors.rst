.. title:: clang-tidy - cppcoreguidelines-avoid-bounds-errors

cppcoreguidelines-avoid-bounds-errors
=====================================

This check enforces the `SL.con.3 <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#slcon3-avoid-bounds-errors>` guideline.
It flags all uses of `operator[]` on `std::vector`, `std::array`, `std::deque`, `std::map`, `std::unordered_map`, and `std::flat_map` and suggests to replace it with `at()`.
Note that `std::span` and `std::mdspan` do not support `at()` as of C++23, so the use of `operator[]` is not flagged.

For example the code

.. code-block:: c++
  std::array<int, 3> a;
  int b = a[4];

will be replaced by 

.. code-block:: c++
  std::vector<int, 3> a;
  int b = a.at(4);
