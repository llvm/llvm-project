.. title:: clang-tidy - readability-redundant-string-init

readability-redundant-string-init
=================================


Finds unnecessary string initializations.

Examples:

.. code:: c++

  // Initializing string with empty string literal is unnecessary.
  std::string a = "";
  std::string b("");
