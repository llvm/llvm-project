.. title:: clang-tidy - readability-redundant-string-init

readability-redundant-string-init
=================================

Finds unnecessary string initializations.

Examples
--------

.. code-block:: c++

  // Initializing string with empty string literal is unnecessary.
  std::string a = "";
  std::string b("");

  // becomes

  std::string a;
  std::string b;

  // Initializing a string_view with an empty string literal produces an
  // instance that compares equal to string_view().
  std::string_view a = "";
  std::string_view b("");

  // becomes
  std::string_view a;
  std::string_view b;

Options
-------

.. option:: StringNames

    Default is `::std::basic_string;::std::basic_string_view`.

    Semicolon-delimited list of class names to apply this check to.
    By default `::std::basic_string` applies to ``std::string`` and
    ``std::wstring``. Set to e.g. `::std::basic_string;llvm::StringRef;QString`
    to perform this check on custom classes.
