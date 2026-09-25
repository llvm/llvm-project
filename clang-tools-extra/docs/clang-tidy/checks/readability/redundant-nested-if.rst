.. title:: clang-tidy - readability-redundant-nested-if

readability-redundant-nested-if
===============================

Finds nested ``if`` statements that can be merged by combining their
conditions with ``&&``.

Example:

.. code-block:: c++

  if (a) {
    if (b) {
      work();
    }
  }

becomes

.. code-block:: c++

  if ((a) && (b)) {
    work();
  }

The check also supports outer declaration conditions in C++17 and later:

.. code-block:: c++

  if (bool X = ready()) {
    if (can_run()) {
      work();
    }
  }

becomes

.. code-block:: c++

  if (bool X = ready(); X && (can_run())) {
    work();
  }

For ``if constexpr``, dependent nested conditions are merged only when they can
be formed outside the discarded branch. This includes conditions such as
non-type template parameters and ``requires`` expressions, but excludes
conditions such as ``sizeof(typename T::type)`` after an earlier dependent
condition.

Options
-------

.. option:: AllowUserDefinedBoolConversion

   When set to `true`, the check also diagnoses chains whose merged conditions
   require user-defined conversion to ``bool``. Fix-its insert
   ``static_cast<bool>(...)`` where needed so the merged condition still uses
   built-in ``&&`` semantics. Default is `false`.
