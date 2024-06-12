.. title:: clang-tidy - cppcoreguidelines-prefer-at-over-subscript-operator

cppcoreguidelines-prefer-at-over-subscript-operator
===================================================

This check flags all uses of ``operator[]`` where an equivalent (same parameter
and return types) ``at()`` method exists and suggest using that instead.

For example the code

.. code-block:: c++
  std::array<int, 3> a;
  int b = a[4];

will generate a warning but 

.. code-block:: c++
  std::unique_ptr<int> a;
  int b = a[0];

will not.

The classes ``std::map``, ``std::unordered_map`` and ``std::flat_map`` are
excluded from this check, because for them the subscript operator has a defined
behaviour when a key does not exist (inserting a new element).

Options
-------

.. option:: ExcludeClasses

    Semicolon-delimited list of class names that should additionally be
    excluded from this check. By default empty.

This check enforces part of the `SL.con.3
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#slcon3-avoid-bounds-errors>`
guideline.
