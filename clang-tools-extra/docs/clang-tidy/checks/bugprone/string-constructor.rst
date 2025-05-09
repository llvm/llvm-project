.. title:: clang-tidy - bugprone-string-constructor

bugprone-string-constructor
===========================

Finds string constructors that are suspicious and probably errors.

A common mistake is to swap parameters to the 'fill' string-constructor.

Examples:

.. code-block:: c++

  std::string str('x', 50); // should be str(50, 'x')

Calling the string-literal constructor with a length bigger than the literal is
suspicious and adds extra random characters to the string.

Examples:

.. code-block:: c++

  std::string("test", 200);   // Will include random characters after "test".
  std::string("test", 2, 5);  // Will include random characters after "st".
  std::string_view("test", 200);

Creating an empty string from constructors with parameters is considered
suspicious. The programmer should use the empty constructor instead.

Examples:

.. code-block:: c++

  std::string("test", 0);   // Creation of an empty string.
  std::string("test", 1, 0);
  std::string_view("test", 0);

Passing an invalid first character position parameter to constructor will
cause ``std::out_of_range`` exception at runtime. 

Examples:

.. code-block:: c++

  std::string("test", -1, 10); // Negative first character position.
  std::string("test", 10, 10); // First character position is bigger than string literal character range".

Options
-------

.. option::  WarnOnLargeLength

   When `true`, the check will warn on a string with a length greater than
   :option:`LargeLengthThreshold`. Default is `true`.

.. option::  LargeLengthThreshold

   An integer specifying the large length threshold. Default is `0x800000`.

.. option:: StringNames

    Default is `::std::basic_string;::std::basic_string_view`.

    Semicolon-delimited list of class names to apply this check to.
    By default `::std::basic_string` applies to ``std::string`` and
    ``std::wstring``. Set to e.g. `::std::basic_string;llvm::StringRef;QString`
    to perform this check on custom classes.
