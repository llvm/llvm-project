.. title:: clang-tidy - performance-prefer-single-char-overloads

performance-prefer-single-char-overloads
========================================

Optimize calls to ``std::string::find()`` and friends when the needle passed is
a single character string literal. The character literal overload is more
efficient.

Examples:

.. code-block:: c++

  str.find("A");
  str += "B";

  // becomes

  str.find('A');
  str += 'B';

This check flags passing strings of size 1 to miscellaneous member functions
as well as ``operator+=``.

Options
-------

.. option:: StringLikeClasses

   Semicolon-separated list of names of string-like classes. By default only
   ``::std::basic_string`` and ``::std::basic_string_view`` are considered.
   Within these classes, the check will only consider member functions named
   ``find``, ``rfind``, ``find_first_of``, ``find_first_not_of``,
   ``find_last_of``, ``find_last_not_of``, ``starts_with``, ``ends_with``,
   ``contains``, or ``operator+=``.
