.. title:: clang-tidy - modernize-use-span

modernize-use-span
==================

This check suggests using ``std::span`` in function parameters when taking a
range of contiguous elements by reference. ``std::span`` can accept vectors,
arrays, and C-style arrays, so the function is no longer bound to a specific
container type.

This check implements `R.14<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rr-ap>``
from the C++ Core Guidelines.

This check requires C++20 or later.

Examples
--------

.. code-block:: c++

  // Before
  void process(const std::vector<int>& vec) {
    for (const auto& val : vec) {
      // Process val
    }
  }

  void analyze(const std::array<double, 5>& arr) {
    for (const auto& val : arr) {
      // Analyze val
    }
  }

  // After
  void process(std::span<const int> vec) {
    for (const auto& val : vec) {
      // Process val
    }
  }

  void analyze(std::span<const double, 5> arr) {
    for (const auto& val : arr) {
      // Analyze val
    }
  }

The transformed code can now accept any contiguous container of the appropriate
element type and optionally specified length.
