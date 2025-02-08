.. title:: clang-tidy - use-cpp-style-comments

readability-use-cpp-style-comments
==================================

Replace C-style comments with C++-style comments.
C-style comments (``/* ... */``) are inherited from C, while C++ introduces 
``//`` as a more concise, readable, and less error-prone alternative. Modern C++ 
guidelines recommend using C++-style comments for consistency and 
maintainability. This check identifies and replaces C-style comments with 
equivalent C++-style comments.

Examples:

Input:

.. code-block::c++

  /* This is a single-line comment */
  int x = 42;  /* Inline comment */

  /* This is a
  multi-line comment */

Output:

.. code-block::c++

  // This is a single-line comment
  int x = 42;  // Inline comment

  // This is a
  // multi-line comment

.. note::

  Inline Comments are neither fixed nor warned.

  Example:
  
  .. code-block:: c++

    int a = /* this is a comment */ 5;

Options
-------

.. option:: ExcludeDoxygenStyleComments

   A boolean option that determines whether Doxygen-style comments should be excluded.  
   By default, this option is set to ``false``.  
