.. title:: clang-tidy - performance-construct-reusable-objects-once

performance-construct-reusable-objects-once
===========================================

Finds variable declarations of expensive-to-construct classes that are
constructed from only constant literals and so can be reused. These objects
should be made ``static``, declared as a global variable or moved to a class
member to avoid repeated construction costs on each function invocation.

This check is particularly useful for identifying inefficiently constructed
regular expressions since their constructors are relatively expensive compared
to their usage, so creating them repeatedly with the same pattern can
significantly impact program performance.

Example:

.. code-block:: c++

  void parse() {
    std::regex r("pattern");  // warning
    // ...
  }

The more efficient version could be any of the following:

.. code-block:: c++

  void parse() {
    static std::regex r("pattern");  // static constructed only once
    // ...
  }

.. code-block:: c++
  
  std::regex r("pattern");  // global variable constructed only once

  void parse() {
    // ...
  }

.. code-block:: c++

  class Parser {
    void parse() {
      // ...
    }

    std::regex r{"pattern"}; // class member constructed only once
  }


Known Limitations
-----------------

The check will not analyze variables that are template dependent.

.. code-block:: c++

  template <typename T>
  void parse() {
    std::basic_regex<T> r("pattern");  // no warning
  }

The check only warns on variables that are constructed from literals or const
variables that are directly constructed from literals.

.. code-block:: c++

  void parse() {
    const int var = 1;
    const int constructed_from_var = var;
    std::regex r1("pattern", var);  // warning
    std::regex r2("pattern", constructed_from_var);  // no warning
  }


Options
-------

.. option:: CheckedClasses

  Semicolon-separated list of fully qualified class names that are considered
  expensive to construct and should be flagged by this check. Default is
  `::std::basic_regex;::boost::basic_regex`.

.. option:: IgnoredFunctions

  Semicolon-separated list of fully qualified function names that are expected
  to be called once so they should not be flagged by this check. Default is
  `::main`.
