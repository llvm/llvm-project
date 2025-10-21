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

``std::all_of``,
``std::any_of``,
``std::binary_search``,
``std::copy``,
``std::copy_if``,
``std::count``,
``std::count_if``,
``std::equal``,
``std::fill``,
``std::find``,
``std::find_if``,
``std::find_if_not``,
``std::for_each``,
``std::includes``,
``std::is_sorted``,
``std::lower_bound``,
``std::max_element``,
``std::min_element``,
``std::mismatch``,
``std::none_of``,
``std::partition``,
``std::partition_point``,
``std::remove_if``,
``std::replace``,
``std::stable_sort``,
``std::transform``,
``std::uninitialized_copy``,
``std::unique``,
``std::upper_bound``.

The check will add the necessary ``#include "llvm/ADT/STLExtras.h"`` directive
when applying fixes.
