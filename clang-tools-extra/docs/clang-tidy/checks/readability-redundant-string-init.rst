.. title:: clang-tidy - readability-redundant-string-init

readability-redundant-string-init
=================================

Finds unnecessary string initializations.

Examples:

.. code-block:: c++

  // Initializing string with empty string literal is unnecessary.
  std::string a = "";
  std::string b("");

  // becomes

  std::string a;
  std::string b;
