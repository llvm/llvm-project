.. title:: clang-tidy - llvm-use-ranges

llvm-use-ranges
===============

Finds calls to STL library iterator algorithms that could be replaced with
LLVM range-based algorithms from ``llvm/ADT/STLExtras.h``.

Example
-------

.. code-block:: c++

  auto it = std::find(vec.begin(), vec.end(), value);
  bool all = std::all_of(vec.begin(), vec.end(),
                         [](int x) { return x > 0; });

Transforms to:

.. code-block:: c++

  auto it = llvm::find(vec, value);
  bool all = llvm::all_of(vec, [](int x) { return x > 0; });

Supported algorithms
--------------------

Calls to the following STL algorithms are checked:

``std::accumulate``,
``std::adjacent_find``,
``std::all_of``,
``std::any_of``,
``std::binary_search``,
``std::copy_if``,
``std::copy``,
``std::count_if``,
``std::count``,
``std::equal``,
``std::fill``,
``std::find_if_not``,
``std::find_if``,
``std::find``,
``std::for_each``,
``std::includes``,
``std::is_sorted``,
``std::lower_bound``,
``std::max_element``,
``std::min_element``,
``std::mismatch``,
``std::none_of``,
``std::partition_point``,
``std::partition``,
``std::remove_if``,
``std::replace_copy_if``,
``std::replace_copy``,
``std::replace``,
``std::search``,
``std::stable_sort``,
``std::transform``,
``std::uninitialized_copy``,
``std::unique``,
``std::upper_bound``.

The check will add the necessary ``#include "llvm/ADT/STLExtras.h"`` directive
when applying fixes.
