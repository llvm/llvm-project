.. title:: clang-tidy - misc-use-anonymous-namespace

misc-use-anonymous-namespace
============================

Finds instances of ``static`` functions or variables declared at global scope
that could instead be moved into an anonymous namespace. It also detects
instances moved to an anonymous namespace that still keep the redundant
``static``.

Anonymous namespaces are the "superior alternative" according to the C++
Standard. ``static`` was proposed for deprecation, but later un-deprecated to
keep C compatibility [1]. ``static`` is an overloaded term with different meanings in
different contexts, so it can create confusion.

Examples:

.. code-block:: c++

  // Bad
  static void foo();
  static int x;

  // Good
  namespace {
    void foo();
    int x;
  } // namespace

.. code-block:: c++

  // Bad
  namespace {
    static void foo();
    static int x;
  }

  // Good
  namespace {
    void foo();
    int x;
  }  // namespace

[1] `Undeprecating static <https://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1012>`_
