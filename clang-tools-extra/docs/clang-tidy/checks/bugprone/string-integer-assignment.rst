.. title:: clang-tidy - bugprone-string-integer-assignment

bugprone-string-integer-assignment
==================================

The check finds assignments of an integer to ``std::basic_string<CharT>``
(``std::string``, ``std::wstring``, etc.). The source of the problem is the
following assignment operator of ``std::basic_string<CharT>``:

.. code-block:: c++

  basic_string& operator=( CharT ch );

Numeric types can be implicitly casted to character types.

.. code-block:: c++

  std::string s;
  int x = 5965;
  s = 6;
  s = x;

Use the appropriate conversion functions or character literals.

.. code-block:: c++

  std::string s;
  int x = 5965;
  s = '6';
  s = std::to_string(x);

In order to suppress false positives, use an explicit cast.

.. code-block:: c++

  std::string s;
  s = static_cast<char>(6);
