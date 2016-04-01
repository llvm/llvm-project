.. title:: clang-tidy - misc-dangling-handle

misc-dangling-handle
====================

Detect dangling references in value handlers like
``std::experimental::string_view``.
These dangling references can come from constructing handles from temporary
values, where the temporary is destroyed soon after the handle is created.

By default only ``std::experimental::basic_string_view`` is considered.
This list can be modified by passing a `;` separated list of class names using
the HandleClasses option.

Examples:

.. code-block:: c++

  string_view View = string();  // View will dangle.
  string A;
  View = A + "A";  // still dangle.

  vector<string_view> V;
  V.push_back(string());  // V[0] is dangling.
  V.resize(3, string());  // V[1] and V[2] will also dangle.

  string_view f() {
    // All these return values will dangle.
    return string();
    string S;
    return S;
    char Array[10]{};
    return Array;
  }
