.. title:: clang-tidy - performance-inefficient-vector-operation

performance-inefficient-vector-operation
========================================

Finds possible inefficient ``std::vector`` operations (e.g. ``push_back``,
``emplace_back``) that may cause unnecessary memory reallocations.

It can also find calls that add element to protobuf repeated field in a loop
without calling Reserve() before the loop. Calling Reserve() first can avoid
unnecessary memory reallocations.

Currently, the check only detects following kinds of loops with a single
statement body:

* Counter-based for loops start with 0:

.. code-block:: c++

  std::vector<int> v;
  for (int i = 0; i < n; ++i) {
    v.push_back(n);
    // This will trigger the warning since the push_back may cause multiple
    // memory reallocations in v. This can be avoid by inserting a 'reserve(n)'
    // statement before the for statement.
  }

  SomeProto p;
  for (int i = 0; i < n; ++i) {
    p.add_xxx(n);
    // This will trigger the warning since the add_xxx may cause multiple memory
    // relloacations. This can be avoid by inserting a
    // 'p.mutable_xxx().Reserve(n)' statement before the for statement.
  }

* For-range loops like ``for (range-declaration : range_expression)``, the type
  of ``range_expression`` can be ``std::vector``, ``std::array``,
  ``std::deque``, ``std::set``, ``std::unordered_set``, ``std::map``,
  ``std::unordered_set``:

.. code-block:: c++

  std::vector<int> data;
  std::vector<int> v;

  for (auto element : data) {
    v.push_back(element);
    // This will trigger the warning since the 'push_back' may cause multiple
    // memory reallocations in v. This can be avoid by inserting a
    // 'reserve(data.size())' statement before the for statement.
  }


Options
-------

.. option:: VectorLikeClasses

   Semicolon-separated list of names of vector-like classes. By default only
   ``::std::vector`` is considered.

.. option:: EnableProto

   When non-zero, the check will also warn on inefficient operations for proto
   repeated fields. Otherwise, the check only warns on inefficient vector
   operations. Default is `0`.
