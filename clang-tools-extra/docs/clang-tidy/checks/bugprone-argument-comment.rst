.. title:: clang-tidy - bugprone-argument-comment

bugprone-argument-comment
=========================

Checks that argument comments match parameter names.

The check understands argument comments in the form ``/*parameter_name=*/``
that are placed right before the argument.

.. code-block:: c++

  void f(bool foo);

  ...

  f(/*bar=*/true);
  // warning: argument name 'bar' in comment does not match parameter name 'foo'

The check tries to detect typos and suggest automated fixes for them.

Options
-------

.. option:: StrictMode

   When zero (default value), the check will ignore leading and trailing
   underscores and case when comparing names -- otherwise they are taken into
   account.

.. option:: CommentBoolLiterals

   When true, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the boolean literal argument.

Before:

.. code-block:: c++

  void foo(bool TurnKey, bool PressButton);

  foo(true, false);

After:

.. code-block:: c++

  void foo(bool TurnKey, bool PressButton);

  foo(/*TurnKey=*/true, /*PressButton=*/false);

.. option:: CommentIntegerLiterals

   When true, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the integer literal argument.

Before:

.. code-block:: c++

  void foo(int MeaningOfLife);

  foo(42);

After:

.. code-block:: c++

  void foo(int MeaningOfLife);

  foo(/*MeaningOfLife=*/42);

.. option:: CommentFloatLiterals

   When true, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the float/double literal argument.

Before:

.. code-block:: c++

  void foo(float Pi);

  foo(3.14159);

After:

.. code-block:: c++

  void foo(float Pi);

  foo(/*Pi=*/3.14159);

.. option:: CommentStringLiterals

   When true, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the string literal argument.

Before:

.. code-block:: c++

  void foo(const char *String);
  void foo(const wchar_t *WideString);

  foo("Hello World");
  foo(L"Hello World");

After:

.. code-block:: c++

  void foo(const char *String);
  void foo(const wchar_t *WideString);

  foo(/*String=*/"Hello World");
  foo(/*WideString=*/L"Hello World");

.. option:: CommentCharacterLiterals

   When true, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the character literal argument.

Before:

.. code-block:: c++

  void foo(char *Character);

  foo('A');

After:

.. code-block:: c++

  void foo(char *Character);

  foo(/*Character=*/'A');

.. option:: CommentUserDefinedLiterals

   When true, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the user defined literal argument.

Before:

.. code-block:: c++

  void foo(double Distance);

  double operator"" _km(long double);

  foo(402.0_km);

After:

.. code-block:: c++

  void foo(double Distance);

  double operator"" _km(long double);

  foo(/*Distance=*/402.0_km);

.. option:: CommentNullPtrs

   When true, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the nullptr literal argument.

Before:

.. code-block:: c++

  void foo(A* Value);

  foo(nullptr);

After:

.. code-block:: c++

  void foo(A* Value);

  foo(/*Value=*/nullptr);
