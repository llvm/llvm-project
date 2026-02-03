.. title:: clang-tidy - performance-inefficient-copy-assign

performance-inefficient-copy-assign
===================================

Suggests insertion of ``std::move(...)`` to turn copy assignment operator calls
into move assignment ones, when deemed valid and profitable.

.. code-block:: c++

  void assign(std::vector<int>& out) {
    std::vector<int> some = make_vector();
    use_vector(some);
    out = some;
  }

  // becomes

  void assign(std::vector<int>& out) {
    std::vector<int> some = make_vector();
    use_vector(some);
    out = std::move(some);
  }
