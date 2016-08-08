.. title:: clang-tidy - modernize-use-bool-literals

modernize-use-bool-literals
===========================

Finds integer literals which are cast to ``bool``.

.. code-block:: c++

  bool p = 1;
  bool f = static_cast<bool>(1);
  std::ios_base::sync_with_stdio(0);
  bool x = p ? 1 : 0;

  // transforms to

  bool p = true;
  bool f = true;
  std::ios_base::sync_with_stdio(false);
  bool x = p ? true : false;
