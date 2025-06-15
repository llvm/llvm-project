.. title:: clang-tidy - readability-use-numeric-limits

readability-use-numeric-limits
==============================

Finds certain integer literals and suggests replacing them with equivalent
``std::numeric_limits`` calls.

Before:

.. code-block:: c++

  void foo() {
    int32_t a = 2147483647;
  }

After:

.. code-block:: c++

  void foo() {
    int32_t a = std::numeric_limits<int32_t>::max();
  }

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.
