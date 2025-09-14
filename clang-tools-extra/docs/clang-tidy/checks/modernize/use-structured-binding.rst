.. title:: clang-tidy - modernize-use-structured-binding

modernize-use-structured-binding
================================

Suggests using C++17 structured bindings to decompose pairs.

This check finds three code patterns and recommends using structured bindings for clearer, more idiomatic C++17 code.

1. Decompose a pair variable by assigning its members to separate variables right after its definition:

.. code-block:: c++

  auto p = getPair<int, int>();
  int x = p.first;
  int y = p.second;

  into:

  auto [x, y] = getPair<int, int>();

2. Use `std::tie` to decompose a pair into two predefined variables:

.. code-block:: c++

  int a;
  int b;
  std::tie(a, b) = getPair<int, int>();

  into:

  auto [a, b] = getPair<int, int>();

3. Manually decompose a pair by assigning to its members to local variables in a range-based for loop:

.. code-block:: c++

  for (autop : vecOfPairs) {
    int x = p.first;
    int y = p.second;
    // ...
  }

  into:

  for (auto [x, y] : vecOfPairs) {
    // use x and y
  }

The check also supports custom pair-like types via the `PairTypes` option.

Options
-------

.. option:: PairTypes

   A Semicolon-separated list of type names to be treated as pair-like for structured binding suggestions.  
   Example: `PairTypes=MyPairType; OtherPairType`. Default is `std::pair`.
