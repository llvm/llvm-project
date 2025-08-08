.. title:: clang-tidy - cppcoreguidelines-pro-bounds-avoid-unchecked-container-access

cppcoreguidelines-pro-bounds-avoid-unchecked-container-access
=============================================================

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

STL containers for which ``operator[]`` is well-defined for all inputs are excluded
from this check (e.g.: ``std::map::operator[]``).

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

    The function to use in the `function` mode. For C++23 and beyond, the
    passed function must support the empty subscript operator, i.e., the case
    where ``a[]`` becomes ``f(a)``. :option:`FixFunctionEmptyArgs` can be
    used to override the suggested function in that case. The default is `gsl::at`. 

.. option:: FixFunctionEmptyArgs

    The function to use in the `function` mode for the empty subscript operator
    case in C++23 and beyond only. If no fixes should be made for empty
    subscript operators, pass an empty string. In that case, only the warnings
    will be printed. The default is the value of :option:`FixFunction`.
