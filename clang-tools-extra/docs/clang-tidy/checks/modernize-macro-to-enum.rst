.. title:: clang-tidy - modernize-macro-to-enum

modernize-macro-to-enum
=======================

Replaces groups of adjacent macros with an unscoped anonymous enum.
Using an unscoped anonymous enum ensures that everywhere the macro
token was used previously, the enumerator name may be safely used.

This check can be used to enforce the C++ core guideline `Enum.1:
Prefer enumerations over macros
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#enum1-prefer-enumerations-over-macros>`_,
within the constraints outlined below.

Potential macros for replacement must meet the following constraints:

- Macros must expand only to integral literal tokens or simple expressions
  of literal tokens.  The unary operators plus, minus and tilde are
  recognized to allow for positive, negative and bitwise negated integers.
  The above expressions may also be surrounded by matching pairs of
  parentheses.  More complicated integral constant expressions are not
  recognized by this check.
- Macros must be defined on sequential source file lines, or with
  only comment lines in between macro definitions.
- Macros must all be defined in the same source file.
- Macros must not be defined within a conditional compilation block.
  (Conditional include guards are exempt from this constraint.)
- Macros must not be defined adjacent to other preprocessor directives.
- Macros must not be used in any conditional preprocessing directive.
- Macros must not be undefined.

Each cluster of macros meeting the above constraints is presumed to
be a set of values suitable for replacement by an anonymous enum.
From there, a developer can give the anonymous enum a name and
continue refactoring to a scoped enum if desired.  Comments on the
same line as a macro definition or between subsequent macro definitions
are preserved in the output.  No formatting is assumed in the provided
replacements, although clang-tidy can optionally format all fixes.

Examples:

.. code-block:: c++

  #define RED   0xFF0000
  #define GREEN 0x00FF00
  #define BLUE  0x0000FF

  #define TM_NONE (-1) // No method selected.
  #define TM_ONE 1    // Use tailored method one.
  #define TM_TWO 2    // Use tailored method two.  Method two
                      // is preferable to method one.
  #define TM_THREE 3  // Use tailored method three.

becomes

.. code-block:: c++

  enum {
  RED = 0xFF0000,
  GREEN = 0x00FF00,
  BLUE = 0x0000FF
  };

  enum {
  TM_NONE = (-1), // No method selected.
  TM_ONE = 1,    // Use tailored method one.
  TM_TWO = 2,    // Use tailored method two.  Method two
                      // is preferable to method one.
  TM_THREE = 3  // Use tailored method three.
  };
