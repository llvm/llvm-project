.. title:: clang-tidy - readability-string-compare

readability-string-compare
==========================

Finds string comparisons using the compare method.

A common mistake is to use the string's ``compare`` method instead of using the
equality or inequality operators. The compare method is intended for sorting
functions and thus returns a negative number, a positive number or
zero depending on the lexicographical relationship between the strings compared.
If an equality or inequality check can suffice, that is recommended. This is
recommended to avoid the risk of incorrect interpretation of the return value
and to simplify the code. The string equality and inequality operators can
also be faster than the ``compare`` method due to early termination.

Examples:

.. code-block:: c++

  std::string str1{"a"};
  std::string str2{"b"};

  // use str1 != str2 instead.
  if (str1.compare(str2)) {
  }

  // use str1 == str2 instead.
  if (!str1.compare(str2)) {
  }

  // use str1 == str2 instead.
  if (str1.compare(str2) == 0) {
  }

  // use str1 != str2 instead.
  if (str1.compare(str2) != 0) {
  }

  // use str1 == str2 instead.
  if (0 == str1.compare(str2)) {
  }

  // use str1 != str2 instead.
  if (0 != str1.compare(str2)) {
  }

  // Use str1 == "foo" instead.
  if (str1.compare("foo") == 0) {
  }

The above code examples show the list of if-statements that this check will
give a warning for. All of them uses ``compare`` to check if equality or
inequality of two strings instead of using the correct operators.
