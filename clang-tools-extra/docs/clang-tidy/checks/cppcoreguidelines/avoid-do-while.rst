.. title:: clang-tidy - cppcoreguidelines-avoid-do-while

cppcoreguidelines-avoid-do-while
================================

Warns when using ``do-while`` loops. They are less readable than plain ``while``
loops, since the termination condition is at the end and the condition is not
checked prior to the first iteration. This can lead to subtle bugs.

This check implements `ES.75
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-do>`_
from the C++ Core Guidelines.

Examples:

.. code-block:: c++

  int x;
  do {
      std::cin >> x;
      // ...
  } while (x < 0);

Options
-------

.. option:: IgnoreMacros

  Ignore the check when analyzing macros. This is useful for safely defining function-like macros:

  .. code-block:: c++

    #define FOO_BAR(x) \
    do { \
      foo(x); \
      bar(x); \
    } while(0)

  Defaults to `false`.
