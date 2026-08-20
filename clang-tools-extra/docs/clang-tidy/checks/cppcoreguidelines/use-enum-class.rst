.. title:: clang-tidy - cppcoreguidelines-use-enum-class

cppcoreguidelines-use-enum-class
================================

Finds unscoped (non-class) ``enum`` declarations and suggests using
``enum class`` instead. Unnamed enum are ignored and will be handled by check implementing `Enum.6
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#enum6-avoid-unnamed-enumerations>`_.

This check implements `Enum.3
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#renum-class>`_
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

Options
-------

.. option:: IgnoreUnscopedEnumsInClasses

   When `true`, ignores unscoped ``enum`` declarations in classes.
   Default is `false`.

.. option:: IgnoreMacros

   When `true`, ignores unscoped ``enum`` declarations within macros.
   Default is `false`.
