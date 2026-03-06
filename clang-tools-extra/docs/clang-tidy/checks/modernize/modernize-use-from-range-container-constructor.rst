.. title:: clang-tidy - modernize-use-from-range-container-constructor

modernize-use-from-range-container-constructor
=================================================

The ``modernize-use-from-range-container-constructor`` check finds container
constructions that use a pair of iterators and replaces them with the more
modern and concise ``std::from_range`` syntax, available in C++23.

This improves readability and leverages modern C++ features for safer and more
expressive code.

.. code:: c++

  std::set<int> s = {1, 2};
  std::vector<int> v(s.begin(), s.end());

  // transforms to:

  #include <ranges>

  std::set<int> s = {1, 2};
  std::vector<int> v(std::from_range, s);

This check handles all standard library containers that support construction
with std::from_range, such as ``std::vector``, ``std::string``, ``std::map``,
and ``std::unordered_set``.

It also recognizes different forms of obtaining iterators, such as
``cbegin()``/``cend()`` and ``std::begin()``/``std::end()``.

Example with ``std::map`` and ``cbegin``/``cend``:

.. code:: c++

  std::vector<std::pair<int, char>> source = {{1, 'a'}, {2, 'b'}};
  std::map<int, char> dest(source.cbegin(), source.cend());

  // transforms to:

  #include <ranges>

  std::vector<std::pair<int, char>> source = {{1, 'a'}, {2, 'b'}};
  std::map<int, char> dest(std::from_range, source);


The check is also able to handle ranges that are behind pointers or smart
pointers.

.. code:: c++

  auto ptr = std::make_unique<std::vector<int>>();
  std::vector<int> v(ptr->begin(), ptr->end());

  // transforms to:

  #include <ranges>

  auto ptr = std::make_unique<std::vector<int>>();
  std::vector<int> v(std::from_range, *ptr);

Limitations
-----------

The warning only triggers when the types of the source and target container
match, even though std::from_range would work in a few extra cases (notably
when the types can be converted implicitly).

.. code:: c++

  std::set<std::string_view> source = {"a"};
  // Attempting to use std::from_range here fails to compile, so no warning.
  std::vector<std::string> dest(source.begin(), source.end());


  std::set<std::string> source = {"a"};
  // std::from_range would compile, but still, no warning.
  std::vector<std::string_view> dest(source.begin(), source.end());

