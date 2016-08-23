.. title:: clang-tidy - misc-suspicious-missing-comma

misc-suspicious-missing-comma
=============================

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
