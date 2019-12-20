.. title:: clang-tidy - performance-inefficient-algorithm

performance-inefficient-algorithm
=================================


Warns on inefficient use of STL algorithms on associative containers.

Associative containers implements some of the algorithms as methods which
should be preferred to the algorithms in the algorithm header. The methods
can take advantage of the order of the elements.

.. code-block:: c++

  std::set<int> s;
  auto it = std::find(s.begin(), s.end(), 43);

  // becomes

  auto it = s.find(43);

.. code-block:: c++

  std::set<int> s;
  auto c = std::count(s.begin(), s.end(), 43);

  // becomes

  auto c = s.count(43);
