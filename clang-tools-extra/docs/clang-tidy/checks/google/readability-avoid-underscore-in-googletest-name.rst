.. title:: clang-tidy - google-readability-avoid-underscore-in-googletest-name

google-readability-avoid-underscore-in-googletest-name
======================================================

Checks whether there are underscores in googletest test suite names and test
names in test macros:

- ``TEST``
- ``TEST_F``
- ``TEST_P``
- ``TYPED_TEST``
- ``TYPED_TEST_P``

The ``FRIEND_TEST`` macro is not included.

For example:

.. code-block:: c++

  TEST(TestSuiteName, Illegal_TestName) {}
  TEST(Illegal_TestSuiteName, TestName) {}

would trigger the check. `Underscores are not allowed`_ in test suite name nor
test names.

The ``DISABLED_`` prefix, which may be used to
`disable test suites and individual tests`_, is removed from the test suite name
and test name before checking for underscores.

This check does not propose any fixes.

.. _Underscores are not allowed: https://google.github.io/googletest/faq.html#why-should-test-suite-names-and-test-names-not-contain-underscore
.. _disable test suites and individual tests: https://google.github.io/googletest/advanced.html#temporarily-disabling-tests
