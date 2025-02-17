.. title:: clang-tidy - readability-use-numeric-limits

readability-use-numeric-limits
==============================

Replaces certain integer literals with ``std::numeric_limits`` calls.

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
