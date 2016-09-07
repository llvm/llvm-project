.. title:: clang-tidy - performance-implicit-cast-in-loop

performance-implicit-cast-in-loop
=================================

This warning appears in a range-based loop with a loop variable of const ref
type where the type of the variable does not match the one returned by the
iterator. This means that an implicit cast has been added, which can for example
result in expensive deep copies.

Example:

.. code-block:: c++

  map<int, vector<string>> my_map;
  for (const pair<int, vector<string>>& p : my_map) {}
  // The iterator type is in fact pair<const int, vector<string>>, which means
  // that the compiler added a cast, resulting in a copy of the vectors.

The easiest solution is usually to use ``const auto&`` instead of writing the type
manually.
