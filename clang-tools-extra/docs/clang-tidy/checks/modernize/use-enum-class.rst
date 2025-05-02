.. title:: clang-tidy - modernize-use-enum-class

modernize-use-enum-class
=============================

Scoped enums (enum class) should be preferred over unscoped enums:
https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Renum-class

Unscoped enums in classes are not reported since it is a well
established pattern to limit the scope of plain enums.

Example:

.. code-block:: c++

  enum E {};        // use "enum class E {};" instead
  enum class E {};  // OK

  struct S {
      enum E {};    // OK, scope already limited
  };

  namespace N {
      enum E {};    // use "enum class E {};" instead
                    // report since it is hard to detect how large the surrounding namespace is
  }
