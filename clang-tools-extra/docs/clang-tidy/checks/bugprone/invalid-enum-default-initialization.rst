.. title:: clang-tidy - bugprone-invalid-enum-default-initialization

bugprone-invalid-enum-default-initialization
============================================

Detect default initialization (to 0) of variables with `enum` type where
the enum has no enumerator with value of 0.

In C++ a default initialization is performed if a variable is initialized with
initializer list or in other implicit ways, and no value is specified at the
initialization. In such cases the value 0 is used for the initialization.
This also applies to enumerations even if it does not have an enumerator with
value 0. In this way a variable with the enum type may contain initially an
invalid value (if the program expects that it contains only the listed
enumerator values).

The checker emits a warning only if an enum variable is default-initialized
(contrary to not initialized) and the enum type does not have an enumerator with
value of 0. The enum type can be scoped or non-scoped enum.

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
