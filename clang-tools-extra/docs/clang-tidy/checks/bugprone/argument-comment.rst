.. title:: clang-tidy - bugprone-argument-comment

bugprone-argument-comment
=========================

Checks that argument comments match parameter names and can optionally add
missing comments for literals, init-lists, and constructed temporaries.

The check understands argument comments in the form ``/*parameter_name=*/``
that are placed right before the argument.

.. code-block:: c++

  void f(bool foo);

  ...

  f(/*bar=*/true);
  // warning: argument name 'bar' in comment does not match parameter name 'foo'

The check tries to detect typos and suggest automated fixes for them. It can
also insert missing comments for configured argument kinds.

Options
-------

.. option:: StrictMode

   When `false`, the check will ignore leading and trailing
   underscores and case when comparing names -- otherwise they are taken into
   account. Default is `false`.

.. option:: IgnoreSingleArgument

   When `true`, the check will ignore the single argument. Default is `false`.

.. option:: CommentAnonymousInitLists

   When `true`, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before anonymous braced-init list arguments
   such as ``{}`` and ``{1, 2, 3}``. Default is `false`.

Before:

.. code-block:: c++

  void foo(const std::vector<int> &Dims);

  foo({});

After:

.. code-block:: c++

  void foo(const std::vector<int> &Dims);

  foo(/*Dims=*/{});

.. option:: CommentBoolLiterals

   When `true`, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the boolean literal argument.
   Default is `false`.

Before:

.. code-block:: c++

  void foo(bool TurnKey, bool PressButton);

  foo(true, false);

After:

.. code-block:: c++

  void foo(bool TurnKey, bool PressButton);

  foo(/*TurnKey=*/true, /*PressButton=*/false);

.. option:: CommentCharacterLiterals

   When `true`, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the character literal argument.
   Default is `false`.

Before:

.. code-block:: c++

  void foo(char *Character);

  foo('A');

After:

.. code-block:: c++

  void foo(char *Character);

  foo(/*Character=*/'A');

.. option:: CommentFloatLiterals

   When `true`, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the float/double literal argument.
   Default is `false`.

Before:

.. code-block:: c++

  void foo(float Pi);

  foo(3.14159);

After:

.. code-block:: c++

  void foo(float Pi);

  foo(/*Pi=*/3.14159);

.. option:: CommentIntegerLiterals

   When `true`, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the integer literal argument.
   Default is `false`.

Before:

.. code-block:: c++

  void foo(int MeaningOfLife);

  foo(42);

After:

.. code-block:: c++

  void foo(int MeaningOfLife);

  foo(/*MeaningOfLife=*/42);

.. option:: CommentNullPtrs

   When `true`, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the nullptr literal argument.
   Default is `false`.

Before:

.. code-block:: c++

  void foo(A* Value);

  foo(nullptr);

After:

.. code-block:: c++

  void foo(A* Value);

  foo(/*Value=*/nullptr);

.. option:: CommentParenthesizedTemporaries

   When `true`, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before explicit temporary constructions such as
   ``Type()`` and ``Type(1, 2, 3)``. Default is `false`.

Before:

.. code-block:: c++

  struct Dims {
    Dims();
    Dims(int, int, int);
  };

  void foo(const Dims &DimsValue);

  foo(Dims());
  foo(Dims(1, 2, 3));

After:

.. code-block:: c++

  struct Dims {
    Dims();
    Dims(int, int, int);
  };

  void foo(const Dims &DimsValue);

  foo(/*DimsValue=*/Dims());
  foo(/*DimsValue=*/Dims(1, 2, 3));

.. option:: CommentStringLiterals

   When `true`, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the string literal argument.
   Default is `false`.

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

.. option:: CommentTypedInitLists

   When `true`, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before typed braced-init list arguments such
   as ``Type{}``. Default is `false`.

Before:

.. code-block:: c++

  void foo(const std::vector<int> &Dims);

  foo(std::vector<int>{});

After:

.. code-block:: c++

  void foo(const std::vector<int> &Dims);

  foo(/*Dims=*/std::vector<int>{});

.. option:: CommentUserDefinedLiterals

   When `true`, the check will add argument comments in the format
   ``/*ParameterName=*/`` right before the user defined literal argument.
   Default is `false`.

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
