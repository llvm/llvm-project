.. title:: clang-tidy - performance-move-const-arg

performance-move-const-arg
==========================

The check warns

- if ``std::move()`` is called with a constant argument,

- if ``std::move()`` is called with an argument of a trivially-copyable type,

- if the result of ``std::move()`` is passed as a const reference argument.

In all three cases, the check will suggest a fix that removes the
``std::move()``.

Here are examples of each of the three cases:

.. code-block:: c++

  const string s;
  return std::move(s);  // Warning: std::move of the const variable has no effect

  int x;
  return std::move(x);  // Warning: std::move of the variable of a trivially-copyable type has no effect

  void f(const string &s);
  string s;
  f(std::move(s));  // Warning: passing result of std::move as a const reference argument; no move will actually happen

Options
-------

.. option:: CheckTriviallyCopyableMove

   If `true`, enables detection of trivially copyable types that do not
   have a move constructor. Default is `true`.
