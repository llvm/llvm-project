.. title:: clang-tidy - readability-redundant-member-init

readability-redundant-member-init
=================================

Finds member initializations that are unnecessary because the same default
constructor would be called if they were not present.

Example
-------

.. code-block:: c++

  // Explicitly initializing the member s is unnecessary.
  class Foo {
  public:
    Foo() : s() {}

  private:
    std::string s;
  };

Options
-------

.. option:: IgnoreBaseInCopyConstructors

    Default is ``0``.

    When non-zero, the check will ignore unnecessary base class initializations
    within copy constructors, since some compilers issue warnings/errors when
    base classes are not explicitly intialized in copy constructors. For example,
    ``gcc`` with ``-Wextra`` or ``-Werror=extra`` issues warning or error
    ``base class 'Bar' should be explicitly initialized in the copy constructor``
    if ``Bar()`` were removed in the following example:

.. code-block:: c++

  // Explicitly initializing member s and base class Bar is unnecessary.
  struct Foo : public Bar {
    // Remove s() below. If IgnoreBaseInCopyConstructors!=0, keep Bar().
    Foo(const Foo& foo) : Bar(), s() {}
    std::string s;
  };

