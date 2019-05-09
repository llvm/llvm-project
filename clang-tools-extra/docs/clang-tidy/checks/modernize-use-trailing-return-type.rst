.. title:: clang-tidy - modernize-use-trailing-return-type

modernize-use-trailing-return-type
==================================

Rewrites function signatures to use a trailing return type
(introduced in C++11). This transformation is purely stylistic.
The return type before the function name is replaced by ``auto``
and inserted after the function parameter list (and qualifiers).

Example
-------

.. code-block:: c++

  int f1();
  inline int f2(int arg) noexcept;
  virtual float f3() const && = delete;

transforms to:

.. code-block:: c++

  auto f1() -> int;
  inline auto f2(int arg) -> int noexcept;
  virtual auto f3() const && -> float = delete;

Known Limitations
-----------------

The following categories of return types cannot be rewritten currently:
* function pointers
* member function pointers
* member pointers
* decltype, when it is the top level expression

Unqualified names in the return type might erroneously refer to different entities after the rewrite.
Preventing such errors requires a full lookup of all unqualified names present in the return type in the scope of the trailing return type location.
This location includes e.g. function parameter names and members of the enclosing class (including all inherited classes).
Such a lookup is currently not implemented.

Given the following piece of code

.. code-block:: c++

  struct Object { long long value; };
  Object f(unsigned Object) { return {Object * 2}; }
  class CC {
    int Object;
    struct Object m();
  };
  Object CC::m() { return {0}; }

a careless rewrite would produce the following output:

.. code-block:: c++

  struct Object { long long value; };
  auto f(unsigned Object) -> Object { return {Object * 2}; } // error
  class CC {
    int Object;
    auto m() -> struct Object;
  };
  auto CC::m() -> Object { return {0}; } // error

This code fails to compile because the Object in the context of f refers to the equally named function parameter.
Similarly, the Object in the context of m refers to the equally named class member.
The check can currently only detect a clash with a function parameter name.
