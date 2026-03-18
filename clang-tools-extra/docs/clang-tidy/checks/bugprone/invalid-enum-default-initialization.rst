.. title:: clang-tidy - bugprone-invalid-enum-default-initialization

bugprone-invalid-enum-default-initialization
============================================

Detects default initialization (to 0) of variables with ``enum`` type where
the enum has no enumerator with value of 0.

In C++ a default initialization is performed if a variable is initialized with
initializer list or in other implicit ways, and no value is specified at the
initialization. In such cases the value 0 is used for the initialization.
This also applies to enumerations even if it does not have an enumerator with
value 0. In this way a variable with the ``enum`` type may contain initially an
invalid value (if the program expects that it contains only the listed
enumerator values).

The check emits a warning only if an ``enum`` variable is default-initialized
(contrary to not initialized) and the ``enum`` does not have an enumerator with
value of 0. The type can be a scoped or non-scoped ``enum``. Unions are not
handled by the check (if it contains a member of enumeration type).

Note that the ``enum`` ``std::errc`` is always ignored because it is expected
to be default initialized, despite not defining an enumerator with the value 0.

.. code-block:: c++

  enum class Enum1: int {
    A = 1,
    B
  };

  enum class Enum0: int {
    A = 0,
    B
  };

  void f() {
    Enum1 X1{}; // warn: 'X1' is initialized to 0
    Enum1 X2 = Enum1(); // warn: 'X2' is initialized to 0
    Enum1 X3; // no warning: 'X3' is not initialized
    Enum0 X4{}; // no warning: type has an enumerator with value of 0
  }

  struct S1 {
    Enum1 A;
    S(): A() {} // warn: 'A' is initialized to 0
  };

  struct S2 {
    int A;
    Enum1 B;
  };

  S2 VarS2{}; // warn: member 'B' is initialized to 0

The check applies to initialization of arrays or structures with initialization
lists in C code too. In these cases elements not specified in the list (and have
enum type) are set to 0.

.. code-block:: c

  enum Enum1 {
    Enum1_A = 1,
    Enum1_B
  };
  struct Struct1 {
    int a;
    enum Enum1 b;
  };

  enum Enum1 Array1[2] = {Enum1_A}; // warn: omitted elements are initialized to 0
  enum Enum1 Array2[2][2] = {{Enum1_A}, {Enum1_A}}; // warn: last element of both nested arrays is initialized to 0
  enum Enum1 Array3[2][2] = {{Enum1_A, Enum1_A}}; // warn: elements of second array are initialized to 0

  struct Struct1 S1 = {1}; // warn: element 'b' is initialized to 0


Options
-------

.. option:: IgnoredEnums

  Semicolon-separated list of regexes specifying enums for which this check won't be
  enforced. Default is `::std::errc`.
