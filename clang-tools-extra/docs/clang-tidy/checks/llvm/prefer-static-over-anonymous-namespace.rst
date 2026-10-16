.. title:: clang-tidy - llvm-prefer-static-over-anonymous-namespace

llvm-prefer-static-over-anonymous-namespace
===========================================

Finds function and variable declarations inside anonymous namespace and
suggests replacing them with ``static`` declarations.

The `LLVM Coding Standards <https://llvm.org/docs/CodingStandards.html#restrict-visibility>`_
recommend keeping anonymous namespaces as small as possible and only use them
for class declarations. For functions and variables the ``static`` specifier
should be preferred for restricting visibility.

For example non-compliant code:

.. code-block:: c++

  namespace {

  class StringSort {
  public:
    StringSort(...)
    bool operator<(const char *RHS) const;
  };

  // warning: place method definition outside of an anonymous namespace
  bool StringSort::operator<(const char *RHS) const {}

  // warning: prefer using 'static' for restricting visibility
  void runHelper() {}

  // warning: prefer using 'static' for restricting visibility
  int myVariable = 42;

  }

Should become:

.. code-block:: c++

  // Small anonymous namespace for class declaration
  namespace {

  class StringSort {
  public:
    StringSort(...)
    bool operator<(const char *RHS) const;
  };

  }

  // placed method definition outside of the anonymous namespace
  bool StringSort::operator<(const char *RHS) const {}

  // used 'static' instead of an anonymous namespace
  static void runHelper() {}

  // used 'static' instead of an anonymous namespace
  static int myVariable = 42;


Options
-------

.. option:: AllowVariableDeclarations

  When `true`, allow variable declarations to be in anonymous namespace.
  Default value is `true`.

.. option:: AllowMemberFunctionsInClass

  When `true`, only methods defined in anonymous namespace outside of the
  corresponding class will be warned. Default value is `true`.
