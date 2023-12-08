.. title:: clang-tidy - bugprone-sizeof-container

bugprone-sizeof-container
=========================

The check finds usages of ``sizeof`` on expressions of STL container types. Most
likely the user wanted to use ``.size()`` instead.

All class/struct types declared in namespace ``std::`` having a const ``size()``
method are considered containers, with the exception of ``std::bitset`` and
``std::array``.

Examples:

.. code-block:: c++

  std::string s;
  int a = 47 + sizeof(s); // warning: sizeof() doesn't return the size of the container. Did you mean .size()?

  int b = sizeof(std::string); // no warning, probably intended.

  std::string array_of_strings[10];
  int c = sizeof(array_of_strings) / sizeof(array_of_strings[0]); // no warning, definitely intended.

  std::array<int, 3> std_array;
  int d = sizeof(std_array); // no warning, probably intended.
