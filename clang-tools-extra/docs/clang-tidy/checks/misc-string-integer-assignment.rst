misc-string-integer-assignment
==============================

The check finds assignments of an integer to ``std::basic_string<CharT>``
(``std::string``, ``std::wstring``, etc.). The source of the problem is the
following assignment operator of ``std::basic_string<CharT>``:

.. code:: c++
  basic_string& operator=( CharT ch );

Numeric types can be implicity casted to character types.

.. code:: c++
  std::string s;
  int x = 5965;
  s = 6;
  s = x;

Use the appropriate conversion functions or character literals.

.. code:: c++
  std::string s;
  int x = 5965;
  s = '6';
  s = std::to_string(x);

In order to suppress false positives, use an explicit cast.

.. code:: c++
  std::string s;
  s = static_cast<char>(6);
