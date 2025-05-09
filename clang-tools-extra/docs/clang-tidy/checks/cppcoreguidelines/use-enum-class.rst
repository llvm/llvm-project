.. title:: clang-tidy - cppcoreguidelines-use-enum-class

cppcoreguidelines-use-enum-class
=============================

Finds plain non-class ``enum`` definitions that could use ``enum class``.

This check implements `Enum.3
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Renum-class>`_
from the C++ Core Guidelines."

Example:

.. code-block:: c++

  enum E {};        // use "enum class E {};" instead
  enum class E {};  // OK

  struct S {
      enum E {};    // use "enum class E {};" instead
                    // OK with option IgnoreUnscopedEnumsInClasses
  };

  namespace N {
      enum E {};    // use "enum class E {};" instead
  }


.. option:: IgnoreUnscopedEnumsInClasses

   When `true` (default is `false`), ignores unscoped ``enum`` declarations in classes.
