.. title:: clang-tidy - performance-faster-string-find

performance-faster-string-find
==============================

Optimize calls to std::string::find() and friends when the needle passed is
a single character string literal.
The character literal overload is more efficient.

By default only `std::basic_string` is considered. This list can be modified by
passing a `;` separated list of class names using the `StringLikeClasses`
option. The methods to consired are fixed, though.

Examples:

.. code-block:: c++

  str.find("A");

  // becomes

  str.find('A');
