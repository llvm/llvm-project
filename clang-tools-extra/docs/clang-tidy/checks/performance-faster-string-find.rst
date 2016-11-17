.. title:: clang-tidy - performance-faster-string-find

performance-faster-string-find
==============================

Optimize calls to ``std::string::find()`` and friends when the needle passed is
a single character string literal. The character literal overload is more
efficient.

Examples:

.. code-block:: c++

  str.find("A");

  // becomes

  str.find('A');

Options
-------

.. option:: StringLikeClasses

   Semicolon-separated list of names of string-like classes. By default only
   ``std::basic_string`` is considered. The list of methods to consired is
   fixed.

