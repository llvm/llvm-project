.. title:: clang-tidy - google-readability-avoid-underscore-in-googletest-name

google-readability-avoid-underscore-in-googletest-name
======================================================

Checks whether there are underscores in googletest test and test case names in
test macros:

- ``TEST``
- ``TEST_F``
- ``TEST_P``
- ``TYPED_TEST``
- ``TYPED_TEST_P``

The ``FRIEND_TEST`` macro is not included.

For example:

.. code-block:: c++

  TEST(TestCaseName, Illegal_TestName) {}
  TEST(Illegal_TestCaseName, TestName) {}

would trigger the check. `Underscores are not allowed`_ in test names nor test
case names.

The ``DISABLED_`` prefix, which may be used to `disable individual tests`_, is
ignored when checking test names, but the rest of the rest of the test name is
still checked.

This check does not propose any fixes.

.. _Underscores are not allowed: https://github.com/google/googletest/blob/master/googletest/docs/faq.md#why-should-test-suite-names-and-test-names-not-contain-underscore
.. _disable individual tests: https://github.com/google/googletest/blob/master/googletest/docs/faq.md#why-should-test-suite-names-and-test-names-not-contain-underscore
