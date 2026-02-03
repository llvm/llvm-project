.. title:: clang-tidy - performance-inefficient-copy-assign

performance-inefficient-copy-assign
===================================


Warns on copy assignment operator recieving an lvalue reference when a
profitable move assignment operator exist and would be used if the lvalue
reference were moved through ``std::move``.

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
