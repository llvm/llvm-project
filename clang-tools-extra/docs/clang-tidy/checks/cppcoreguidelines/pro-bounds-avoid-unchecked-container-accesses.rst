.. title:: clang-tidy - cppcoreguidelines-pro-bounds-avoid-unchecked-container-accesses

cppcoreguidelines-pro-bounds-avoid-unchecked-container-accesses
===============================================================

Flags calls to ``operator[]`` in STL containers and suggests replacing it with
safe alternatives.

For example the code

.. code-block:: c++

  std::vector<int> a;
  int b = a[4];

will generate a warning but 

.. code-block:: c++

  std::unique_ptr<vector> a;
  int b = a[0];

will generate a warning.

STL containers with well-defined behavior for ``operator[]`` are excluded from this
check.

This check enforces part of the `SL.con.3
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#slcon3-avoid-bounds-errors>`
guideline and is part of the `Bounds Safety (Bounds 4)
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Pro-bounds-arrayindex>`
profile from the C++ Core Guidelines.

Options
-------

.. option:: SubscriptExcludeClasses

    Semicolon-delimited list of class names that should additionally be
    excluded from this check. The default is an empty string.

.. option:: SubscriptFixMode

    Determines what fixes are suggested. Either `None` (default), `at` (use 
    ``a.at(index)`` if a fitting function exists) or `function` (use a 
    function ``f(a, index)``).

.. option:: SubscriptFixFunction

    The function to use in the `function` mode. ``gsl::at`` by default.
