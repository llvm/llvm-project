.. title:: clang-tidy - readability-isolate-declaration

readability-isolate-declaration
===============================

Detects local variable declarations declaring more than one variable and
tries to refactor the code to one statement per declaration.

The automatic code-transformation will use the same indentation as the original
for every created statement and add a line break after each statement.
It keeps the order of the variable declarations consistent, too.

.. code-block:: c++

  void f() {
    int * pointer = nullptr, value = 42, * const const_ptr = &value;
    // This declaration will be diagnosed and transformed into:
    // int * pointer = nullptr;
    // int value = 42;
    // int * const const_ptr = &value;
  }


The check excludes places where it is necessary or common to declare
multiple variables in one statement and there is no other way supported in the
language. Please note that structured bindings are not considered.

.. code-block:: c++

  // It is not possible to transform this declaration and doing the declaration
  // before the loop will increase the scope of the variable 'Begin' and 'End'
  // which is undesirable.
  for (int Begin = 0, End = 100; Begin < End; ++Begin);
  if (int Begin = 42, Result = some_function(Begin); Begin == Result);

  // It is not possible to transform this declaration because the result is
  // not functionality preserving as 'j' and 'k' would not be part of the
  // 'if' statement anymore.
  if (SomeCondition())
    int i = 42, j = 43, k = function(i,j);


Limitations
-----------

Global variables and member variables are excluded.

The check currently does not support the automatic transformation of
member-pointer-types.

.. code-block:: c++

  struct S {
    int a;
    const int b;
    void f() {}
  };

  void f() {
    // Only a diagnostic message is emitted
    int S::*p = &S::a, S::*const q = &S::a;
  }

Furthermore, the transformation is very cautious when it detects various kinds
of macros or preprocessor directives in the range of the statement. In this
case the transformation will not happen to avoid unexpected side-effects due to
macros.

.. code-block:: c++

  #define NULL 0
  #define MY_NICE_TYPE int **
  #define VAR_NAME(name) name##__LINE__
  #define A_BUNCH_OF_VARIABLES int m1 = 42, m2 = 43, m3 = 44;

  void macros() {
    int *p1 = NULL, *p2 = NULL;
    // Will be transformed to
    // int *p1 = NULL;
    // int *p2 = NULL;

    MY_NICE_TYPE p3, v1, v2;
    // Won't be transformed, but a diagnostic is emitted.

    int VAR_NAME(v3),
        VAR_NAME(v4),
        VAR_NAME(v5);
    // Won't be transformed, but a diagnostic is emitted.

    A_BUNCH_OF_VARIABLES
    // Won't be transformed, but a diagnostic is emitted.

    int Unconditional,
  #if CONFIGURATION
        IfConfigured = 42,
  #else
        IfConfigured = 0;
  #endif
    // Won't be transformed, but a diagnostic is emitted.
  }
