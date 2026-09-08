.. title:: clang-tidy - readability-trailing-comma

readability-trailing-comma
==========================

Checks for presence or absence of trailing commas in enum definitions and
initializer lists.

The check supports separate policies for single-line and multi-line constructs,
allowing different styles for each. By default, the check enforces trailing
commas in multi-line constructs and removes them from single-line constructs.

Trailing commas in multi-line constructs offer several benefits:

- Adding or removing elements at the end only changes a single line, making
  diffs smaller and easier to read.
- Formatters may change code to a more desired style.
- Code generators avoid the need for special handling of the last element.

.. code-block:: c++

  // Without trailing commas - adding "Yellow" requires modifying the "Blue" line
  enum Color {
    Red,
    Green,
    Blue
  };

  // With trailing commas - adding "Yellow" is a clean, single-line change
  enum Color {
    Red,
    Green,
    Blue,
  };


Limitations
-----------

The check currently doesn't analyze code inside macros.


Options
-------

.. option:: SingleLineCommaPolicy

  Controls whether to add, remove, or ignore trailing commas in single-line
  enum definitions and initializer lists.
  Valid values are:

  - `Append`: Add trailing commas where missing.
  - `Remove`: Remove trailing commas where present.
  - `Ignore`: Do not check single-line constructs.

  Default is `Remove`.

.. option:: MultiLineCommaPolicy

  Controls whether to add, remove, or ignore trailing commas in multi-line
  enum definitions and initializer lists.
  Valid values are:

  - `Append`: Add trailing commas where missing.
  - `Remove`: Remove trailing commas where present.
  - `Ignore`: Do not check multi-line constructs.

  Default is `Append`.
