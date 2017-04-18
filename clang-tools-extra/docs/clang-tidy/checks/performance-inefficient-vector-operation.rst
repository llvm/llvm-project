.. title:: clang-tidy - performance-inefficient-vector-operation

performance-inefficient-vector-operation
========================================

Finds possible inefficient `std::vector` operations (e.g. `push_back`) that may
cause unnecessary memory reallocations.

Currently, the check only detects a typical counter-based for loop with a single
statement in it, see below:

.. code-block:: c++

  std::vector<int> v;
  for (int i = 0; i < n; ++i) {
    v.push_back(n);
    // This will trigger the warning since the push_back may cause multiple
    // memory reallocations in v. This can be avoid by inserting a 'reserve(n)'
    // statment before the for statment.
  }
