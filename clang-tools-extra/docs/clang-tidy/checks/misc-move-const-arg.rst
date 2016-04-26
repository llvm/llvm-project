.. title:: clang-tidy - misc-move-const-arg

misc-move-const-arg
===================

The check warns if ``std::move()`` is called with a constant argument or an
argument of a trivially-copyable type, e.g.:

.. code:: c++

  const string s;
  return std::move(s);  // Warning: std::move of the const variable has no effect

  int x;
  return std::move(x);  // Warning: std::move of the variable of a trivially-copyable type has no effect
