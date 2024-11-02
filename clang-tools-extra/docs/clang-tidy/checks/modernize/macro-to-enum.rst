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

- Macros must expand only to integral literal tokens or expressions
  of literal tokens.  The expression may contain any of the unary
  operators ``-``, ``+``, ``~`` or ``!``, any of the binary operators
  ``,``, ``-``, ``+``, ``*``, ``/``, ``%``, ``&``, ``|``, ``^``, ``<``,
  ``>``, ``<=``, ``>=``, ``==``, ``!=``, ``||``, ``&&``, ``<<``, ``>>``
  or ``<=>``, the ternary operator ``?:`` and its
  `GNU extension <https://gcc.gnu.org/onlinedocs/gcc/Conditionals.html>`_.
  Parenthesized expressions are also recognized.  This recognizes
  most valid expressions.  In particular, expressions with the
  ``sizeof`` operator are not recognized.
- Macros must be defined on sequential source file lines, or with
  only comment lines in between macro definitions.
- Macros must all be defined in the same source file.
- Macros must not be defined within a conditional compilation block.
  (Conditional include guards are exempt from this constraint.)
- Macros must not be defined adjacent to other preprocessor directives.
- Macros must not be used in any conditional preprocessing directive.
- Macros must not be used as arguments to other macros.
- Macros must not be undefined.
- Macros must be defined at the top-level, not inside any declaration or
  definition.

Each cluster of macros meeting the above constraints is presumed to
be a set of values suitable for replacement by an anonymous enum.
From there, a developer can give the anonymous enum a name and
continue refactoring to a scoped enum if desired.  Comments on the
same line as a macro definition or between subsequent macro definitions
are preserved in the output.  No formatting is assumed in the provided
replacements, although clang-tidy can optionally format all fixes.

.. warning::

  Initializing expressions are assumed to be valid initializers for
  an enum.  C requires that enum values fit into an ``int``, but
  this may not be the case for some accepted constant expressions.
  For instance ``1 << 40`` will not fit into an ``int`` when the size of
  an ``int`` is 32 bits.

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
