.. title:: clang-tidy - readability-redundant-declaration

readability-redundant-declaration
=================================

Finds redundant variable and function declarations.

.. code-block:: c++

  extern int X;
  extern int X;

becomes

.. code-block:: c++

  extern int X;

Such redundant declarations can be removed without changing program behavior.
They can for instance be unintentional left overs from previous refactorings
when code has been moved around. Having redundant declarations could in worst
case mean that there are typos in the code that cause bugs.

Normally the code can be automatically fixed, :program:`clang-tidy` can remove
the second declaration. However there are 2 cases when you need to fix the code
manually:

* When the declarations are in different header files;
* When multiple variables are declared together.

Options
-------

.. option:: IgnoreMacros

   If set to `true`, the check will not give warnings inside macros. Default
   is `true`.
