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


Transforms to:

.. code-block:: c++

  auto Iter1 = std::ranges::find(Items, 0);
  auto AreSame = std::ranges::equal(Items1, Items2);

Supported algorithms
--------------------

Calls to the following std library algorithms are checked:

``std::adjacent_find``,
``std::all_of``,
``std::any_of``,
``std::binary_search``,
``std::copy_backward``,
``std::copy_if``,
``std::copy``,
``std::destroy``,
``std::equal_range``,
``std::equal``,
``std::fill``,
``std::find_end``,
``std::find_if_not``,
``std::find_if``,
``std::find``,
``std::for_each``,
``std::generate``,
``std::includes``,
``std::inplace_merge``,
``std::iota``,
``std::is_heap_until``,
``std::is_heap``,
``std::is_partitioned``,
``std::is_permutation``,
``std::is_sorted_until``,
``std::is_sorted``,
``std::lexicographical_compare``,
``std::lower_bound``,
``std::make_heap``,
``std::max_element``,
``std::merge``,
``std::min_element``,
``std::minmax_element``,
``std::mismatch``,
``std::move_backward``,
``std::move``,
``std::next_permutation``,
``std::none_of``,
``std::partial_sort_copy``,
``std::partition_copy``,
``std::partition_point``,
``std::partition``,
``std::pop_heap``,
``std::prev_permutation``,
``std::push_heap``,
``std::remove_copy_if``,
``std::remove_copy``,
``std::remove``, ``std::remove_if``,
``std::replace_if``,
``std::replace``,
``std::reverse_copy``,
``std::reverse``,
``std::rotate``,
``std::rotate_copy``,
``std::sample``,
``std::search``,
``std::set_difference``,
``std::set_intersection``,
``std::set_symmetric_difference``,
``std::set_union``,
``std::shift_left``,
``std::shift_right``,
``std::sort_heap``,
``std::sort``,
``std::stable_partition``,
``std::stable_sort``,
``std::transform``,
``std::uninitialized_copy``,
``std::uninitialized_default_construct``,
``std::uninitialized_fill``,
``std::uninitialized_move``,
``std::uninitialized_value_construct``,
``std::unique_copy``,
``std::unique``,
``std::upper_bound``.

Note: some range algorithms for ``vector<bool>`` require C++23 because it uses
proxy iterators.

Reverse Iteration
-----------------

If calls are made using reverse iterators on containers, The code will be
fixed using the ``std::views::reverse`` adaptor.

.. code-block:: c++
  
  auto AreSame = std::equal(Items1.rbegin(), Items1.rend(),
                            std::crbegin(Items2), std::crend(Items2));

Transforms to:

.. code-block:: c++

  auto AreSame = std::ranges::equal(std::ranges::reverse_view(Items1),
                                    std::ranges::reverse_view(Items2));

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.

.. option:: UseReversePipe

  When `true` (default `false`), fixes which involve reverse ranges will use the
  pipe adaptor syntax instead of the function syntax.

  .. code-block:: c++

    std::find(Items.rbegin(), Items.rend(), 0);

  Transforms to:

  .. code-block:: c++

    std::ranges::find(Items | std::views::reverse, 0);
