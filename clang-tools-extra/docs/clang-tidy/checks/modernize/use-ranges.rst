.. title:: clang-tidy - modernize-use-ranges

modernize-use-ranges
====================

Detects calls to standard library iterator algorithms that could be replaced
with a ranges version instead.

Example
-------

.. code-block:: c++

  auto Iter1 = std::find(Items.begin(), Items.end(), 0);
  auto AreSame = std::equal(Items1.cbegin(), Items1.cend(),
                            std::begin(Items2), std::end(Items2));


transforms to:

.. code-block:: c++

  auto Iter1 = std::ranges::find(Items, 0);
  auto AreSame = std::ranges::equal(Items1, Items2);

Calls to the following std library algorithms are checked:
``::std::all_of``,``::std::any_of``,``::std::none_of``,``::std::for_each``,
``::std::find``,``::std::find_if``,``::std::find_if_not``,
``::std::adjacent_find``,``::std::copy``,``::std::copy_if``,
``::std::copy_backward``,``::std::move``,``::std::move_backward``,
``::std::fill``,``::std::transform``,``::std::replace``,``::std::replace_if``,
``::std::generate``,``::std::remove``,``::std::remove_if``,
``::std::remove_copy``,``::std::remove_copy_if``,``::std::unique``,
``::std::unique_copy``,``::std::sample``,``::std::partition_point``,
``::std::lower_bound``,``::std::upper_bound``,``::std::equal_range``,
``::std::binary_search``,``::std::push_heap``,``::std::pop_heap``,
``::std::make_heap``,``::std::sort_heap``,``::std::next_permutation``,
``::std::prev_permutation``,``::std::iota``,``::std::reverse``,
``::std::reverse_copy``,``::std::shift_left``,``::std::shift_right``,
``::std::is_partitioned``,``::std::partition``,``::std::partition_copy``,
``::std::stable_partition``,``::std::sort``,``::std::stable_sort``,
``::std::is_sorted``,``::std::is_sorted_until``,``::std::is_heap``,
``::std::is_heap_until``,``::std::max_element``,``::std::min_element``,
``::std::minmax_element``,``::std::uninitialized_copy``,
``::std::uninitialized_fill``,``::std::uninitialized_move``,
``::std::uninitialized_default_construct``,
``::std::uninitialized_value_construct``,``::std::destroy``,
``::std::partial_sort_copy``,``::std::includes``,
``::std::set_union``,``::std::set_intersection``,``::std::set_difference``,
``::std::set_symmetric_difference``,``::std::merge``,
``::std::lexicographical_compare``,``::std::find_end``,``::std::search``,
``::std::is_permutation``,``::std::equal``,``::std::mismatch``.

Reverse Iteration
-----------------

If calls are made using reverse iterators on containers, The code will be
fixed using the ``std::views::reverse`` adaptor.

.. code-block:: c++
  
  auto AreSame = std::equal(Items1.rbegin(), Items1.rend(),
                            std::crbegin(Items2), std::crend(Items2));

transformst to:

.. code-block:: c++

  auto AreSame = std::equal(std::views::reverse(Items1),
                            std::views::reverse(Items2));

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.

