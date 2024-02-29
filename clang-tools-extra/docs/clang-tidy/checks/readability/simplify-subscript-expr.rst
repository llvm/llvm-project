.. title:: clang-tidy - readability-simplify-subscript-expr

readability-simplify-subscript-expr
===================================

This check simplifies subscript expressions. Currently this covers calling
``.data()`` and immediately doing an array subscript operation to obtain a
single element, in which case simply calling ``operator[]`` suffice.

Examples:

.. code-block:: c++

  std::string s = ...;
  char c = s.data()[i];  // char c = s[i];

Options
-------

.. option:: Types

   The list of type(s) that triggers this check. Default is
   `::std::basic_string;::std::basic_string_view;::std::vector;::std::array;::std::span`
