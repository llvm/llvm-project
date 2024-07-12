.. title:: clang-tidy - boost-use-ranges

boost-use-ranges
================

Detects calls to standard library iterator algorithms that could be replaced
with a Boost ranges version instead.

Example
-------

.. code-block:: c++

  auto Iter1 = std::find(Items.begin(), Items.end(), 0);
  auto AreSame = std::equal(Items1.cbegin(), Items1.cend(), std::begin(Items2),
                            std::end(Items2));


transforms to:

.. code-block:: c++

  auto Iter1 = boost::range::find(Items, 0);
  auto AreSame = boost::range::equal(Items1, Items2);

Calls to the following std library algorithms are checked:
``includes``,``set_union``,``set_intersection``,``set_difference``,
``set_symmetric_difference``,``unique``,``lower_bound``,``stable_sort``,
``equal_range``,``remove_if``,``sort``,``random_shuffle``,``remove_copy``,
``stable_partition``,``remove_copy_if``,``count``,``copy_backward``,
``reverse_copy``,``adjacent_find``,``remove``,``upper_bound``,``binary_search``,
``replace_copy_if``,``for_each``,``generate``,``count_if``,``min_element``,
``reverse``,``replace_copy``,``fill``,``unique_copy``,``transform``,``copy``,
``replace``,``find``,``replace_if``,``find_if``,``partition``,``max_element``,
``find_end``,``merge``,``partial_sort_copy``,``find_first_of``,``search``,
``lexicographical_compare``,``equal``,``mismatch``,``next_permutation``,
``prev_permutation``,``push_heap``,``pop_heap``,``make_heap``,``sort_heap``,
``copy_if``,``is_permutation``,``is_partitioned``,``find_if_not``,
``partition_copy``,``any_of``,``iota``,``all_of``,``partition_point``,
``is_sorted``,``none_of``,``is_sorted_until``,``reduce``,``accumulate``,
``parital_sum``,``adjacent_difference``.

The check will also look for the following functions from the
``boost::algorithm`` namespace:
``reduce``,``find_backward``,``find_not_backward``,``find_if_backward``,
``find_if_not_backward``,``hex``,``hex_lower``,``unhex``,
``is_partitioned_until``,``is_palindrome``,``copy_if``,``copy_while``,
``copy_until``,``copy_if_while``,``copy_if_until``,``is_permutation``,
``is_partitioned``,``one_of``,``one_of_equal``,``find_if_not``,
``partition_copy``,``any_of``,``any_of_equal``,``iota``,``all_of``,
``all_of_equal``,``partition_point``,``is_sorted_until``,``is_sorted``,
``is_increasing``,``is_decreasing``,``is_strictly_increasing``,
``is_strictly_decreasing``,``none_of``,``none_of_equal``,``clamp_range``,
``apply_permutation``,``apply_reverse_permutation``.

Reverse Iteration
-----------------

If calls are made using reverse iterators on containers, The code will be
fixed using the ``boost::adaptors::reverse`` adaptor.

.. code-block:: c++
  
  auto AreSame = std::equal(Items1.rbegin(), Items1.rend(),
                            std::crbegin(Items2), std::crend(Items2));

transformst to:

.. code-block:: c++

  auto AreSame = std::equal(boost::adaptors::reverse(Items1),
                            boost::adaptors::reverse(Items2));

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.

.. option:: IncludeBoostSystem
   
   If `true` (default value) the boost headers are included as system headers
   with angle brackets (`#include <boost.hpp>`), otherwise quotes are used
   (`#include "boost.hpp"`).
