.. title:: clang-tidy - modernize-use-structured-binding

modernize-use-structured-binding
================================

Finds places where structured bindings could be used to decompose pairs and
suggests replacing them.

This check finds three code patterns and recommends using structured bindings
for clearer, more idiomatic C++17 code.

1. Decompose a pair variable by assigning its members to separate variables
right after its definition:

.. code-block:: c++

  auto p = getPair<int, int>();
  int x = p.first;
  int y = p.second;

  into:

  auto [x, y] = getPair<int, int>();

2. Use ``std::tie`` to decompose a pair into two predefined variables:

.. code-block:: c++

  int a;
  int b;
  std::tie(a, b) = getPair<int, int>();

  into:

  auto [a, b] = getPair<int, int>();

3. Manually decompose a pair by assigning to its members to local variables
in a range-based for loop:

.. code-block:: c++

  for (auto p : vecOfPairs) {
    int x = p.first;
    int y = p.second;
    // ...
  }

  into:

  for (auto [x, y] : vecOfPairs) {
    // use x and y
  }

Limitations
-----------

The check currently ignores variables defined with attributes or qualifiers
except const and & since it's not very common:

.. code-block:: c++

  static auto pair = getPair();
  static int b = pair.first;
  static int c = pair.second;

The check doesn't check for some situations which could possibly be transferred
to structured bindings, for example:

.. code-block:: c++

  const auto& results = mapping.try_emplace("hello!");
  const iterator& it = results.first;
  bool succeed = results.second;
  // succeed is not changed in the following code

and:

.. code-block:: c++

  const auto results = mapping.try_emplace("hello!");
  if (results.second) {
      handle_inserted(results.first);
  }
