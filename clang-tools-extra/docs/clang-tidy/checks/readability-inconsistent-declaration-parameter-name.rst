.. title:: clang-tidy - readability-inconsistent-declaration-parameter-name

readability-inconsistent-declaration-parameter-name
===================================================

Find function declarations which differ in parameter names.

Example:

.. code-block:: c++

  // in foo.hpp:
  void foo(int a, int b, int c);

  // in foo.cpp:
  void foo(int d, int e, int f); // warning

This check should help to enforce consistency in large projects, where it often
happens that a definition of function is refactored, changing the parameter
names, but its declaration in header file is not updated. With this check, we
can easily find and correct such inconsistencies, keeping declaration and
definition always in sync.

Unnamed parameters are allowed and are not taken into account when comparing
function declarations, for example:

.. code-block:: c++

  void foo(int a);
  void foo(int); // no warning

To help with refactoring, in some cases fix-it hints are generated to align
parameter names to a single naming convention. This works with the assumption
that the function definition is the most up-to-date version, as it directly
references parameter names in its body. Example:

.. code-block:: c++

  void foo(int a); // warning and fix-it hint (replace "a" to "b")
  int foo(int b) { return b + 2; } // definition with use of "b"

In the case of multiple redeclarations or function template specializations,
a warning is issued for every redeclaration or specialization inconsistent with
the definition or the first declaration seen in a translation unit.

.. option:: IgnoreMacros

   If this option is set to non-zero (default is `1`), the check will not warn
   about names declared inside macros.
