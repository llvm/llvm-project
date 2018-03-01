.. title:: clang-tidy - bugprone-suspicious-missing-comma

bugprone-suspicious-missing-comma
=================================

String literals placed side-by-side are concatenated at translation phase 6
(after the preprocessor). This feature is used to represent long string
literal on multiple lines.

For instance, the following declarations are equivalent:

.. code-block:: c++

  const char* A[] = "This is a test";
  const char* B[] = "This" " is a "    "test";

A common mistake done by programmers is to forget a comma between two string
literals in an array initializer list.

.. code-block:: c++

  const char* Test[] = {
    "line 1",
    "line 2"     // Missing comma!
    "line 3",
    "line 4",
    "line 5"
  };

The array contains the string "line 2line3" at offset 1 (i.e. Test[1]). Clang
won't generate warnings at compile time.

This check may warn incorrectly on cases like:

.. code-block:: c++

  const char* SupportedFormat[] = {
    "Error %s",
    "Code " PRIu64,   // May warn here.
    "Warning %s",
  };

Options
-------

.. option::  SizeThreshold

   An unsigned integer specifying the minimum size of a string literal to be
   considered by the check. Default is `5U`.

.. option::  RatioThreshold

   A string specifying the maximum threshold ratio [0, 1.0] of suspicious string
   literals to be considered. Default is `".2"`.

.. option::  MaxConcatenatedTokens

   An unsigned integer specifying the maximum number of concatenated tokens.
   Default is `5U`.
