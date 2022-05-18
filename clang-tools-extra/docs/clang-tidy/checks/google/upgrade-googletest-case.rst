.. title:: clang-tidy - google-upgrade-googletest-case

google-upgrade-googletest-case
==============================

Finds uses of deprecated Google Test version 1.9 APIs with names containing
``case`` and replaces them with equivalent APIs with ``suite``.

All names containing ``case`` are being replaced to be consistent with the
meanings of "test case" and "test suite" as used by the International
Software Testing Qualifications Board and ISO 29119.

The new names are a part of Google Test version 1.9 (release pending). It is
recommended that users update their dependency to version 1.9 and then use this
check to remove deprecated names.

The affected APIs are:

- Member functions of ``testing::Test``, ``testing::TestInfo``,
  ``testing::TestEventListener``, ``testing::UnitTest``, and any type inheriting
  from these types
- The macros ``TYPED_TEST_CASE``, ``TYPED_TEST_CASE_P``,
  ``REGISTER_TYPED_TEST_CASE_P``, and ``INSTANTIATE_TYPED_TEST_CASE_P``
- The type alias ``testing::TestCase``

Examples of fixes created by this check:

.. code-block:: c++

  class FooTest : public testing::Test {
  public:
    static void SetUpTestCase();
    static void TearDownTestCase();
  };

  TYPED_TEST_CASE(BarTest, BarTypes);

becomes

.. code-block:: c++

  class FooTest : public testing::Test {
  public:
    static void SetUpTestSuite();
    static void TearDownTestSuite();
  };

  TYPED_TEST_SUITE(BarTest, BarTypes);

For better consistency of user code, the check renames both virtual and
non-virtual member functions with matching names in derived types. The check
tries to provide only a warning when a fix cannot be made safely, as is the case
with some template and macro uses.
