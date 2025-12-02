.. title:: clang-tidy - readability-redundant-parentheses

readability-redundant-parentheses
=================================

Detect redundant parentheses.

When modifying code, one often forgets to remove the corresponding parentheses.
This results in overly lengthy code. When the expression is complex, finding
the matching parentheses becomes particularly difficult.

Example
-------

.. code-block:: c++

  (1);
  ((a + 2)) * 3;
  (a);
  ("aaa");

Currently this check does not take into account the precedence of operations.
Even if the expression within the parentheses has a higher priority than that
outside the parentheses. In other words, removing the parentheses will not
affect the semantics.

.. code-block:: c++

  int a = (1 * 2) + 3; // no warning

Options
-------

.. option:: AllowedDecls

  Semicolon-separated list of regular expressions matching names of declarations
  to ignore when the parentheses are around. Declarations can include variables
  or functions. The default is an `std::max;std::min`.

  Some STL library functions may have the same name as widely used function-like
  macro. For example, ``std::max`` and ``max`` macro. A workaround to distinguish
  them is adding parentheses around functions to prevent function-like macro.
