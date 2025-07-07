.. title:: clang-tidy - cppcoreguidelines-pro-bounds-avoid-unchecked-container-accesses

cppcoreguidelines-pro-bounds-avoid-unchecked-container-accesses
===============================================================

Finds calls to ``operator[]`` in STL containers and suggests replacing them
with safe alternatives.
Safe alternatives include STL ``at`` or GSL ``at`` functions, ``begin()`` or
``end()`` functions, ``range-for`` loops, ``std::span``, or an appropriate
function from ``<algorithms>``.

For example, both

.. code-block:: c++

  std::vector<int> a;
  int b = a[4];

and

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

.. option:: ExcludeClasses

    Semicolon-delimited list of class names for overwriting the default
    exclusion list. The default is:
    `::std::map;::std::unordered_map;::std::flat_map`.
    
.. option:: FixMode

    Determines what fixes are suggested. Either `none`, `at` (use 
    ``a.at(index)`` if a fitting function exists) or `function` (use a 
    function ``f(a, index)``). The default is `none`.

.. option:: FixFunction

    The function to use in the `function` mode. `gsl::at` by default.
