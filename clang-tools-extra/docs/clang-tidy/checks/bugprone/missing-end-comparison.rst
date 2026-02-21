.. title:: clang-tidy - bugprone-missing-end-comparison

bugprone-missing-end-comparison
===============================

Finds instances where the result of a standard algorithm is used in a Boolean
context without being compared to the end iterator.

Standard algorithms such as ``std::find``, ``std::search``, and
``std::lower_bound`` return an iterator to the element if found, or the end
iterator otherwise.

Using the result directly in a Boolean context (like an ``if`` statement) is
almost always a bug, as it only checks if the iterator itself evaluates to
``true``, which may always be true for many iterator types.

Examples:

.. code-block:: c++

  int arr[] = {1, 2, 3};
  int* begin = std::begin(arr);
  int* end = std::end(arr);

  // Problematic:
  if (std::find(begin, end, 2)) {
    // ...
  }

  // Fixed by the check:
  if ((std::find(begin, end, 2) != end)) {
    // ...
  }

  // C++20 ranges:
  std::vector<int> v = {1, 2, 3};
  if (std::ranges::find(v, 2)) { // Problematic
    // ...
  }

  // Fixed by the check:
  if ((std::ranges::find(v, 2) != std::ranges::end(v))) {
    // ...
  }

The check also handles range-based algorithms introduced in C++20.

Supported algorithms:

- ``std::find``
- ``std::find_if``
- ``std::find_if_not``
- ``std::search``
- ``std::search_n``
- ``std::find_end``
- ``std::find_first_of``
- ``std::lower_bound``
- ``std::upper_bound``
- ``std::partition_point``
- ``std::min_element``
- ``std::max_element``
- ``std::adjacent_find``
- ``std::is_sorted_until``
- ``std::ranges::find``
- ``std::ranges::find_if``
- ``std::ranges::find_if_not``
- ``std::ranges::lower_bound``
- ``std::ranges::upper_bound``
- ``std::ranges::min_element``
- ``std::ranges::max_element``
