.. title:: clang-tidy - readability-trailing-comma

readability-trailing-comma
==========================

Checks for presence or absence of trailing commas in enum definitions and
initializer lists.

The check can either append trailing commas where they are missing or remove
them where they are present, based on the configured policy.

Trailing commas offer several benefits:

- Adding or removing elements may only changes a single line, making diffs
  smaller and easier to read.
- Formatters may change code to a more desired style.
- Code generators avoid for need special handling of the last element.

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

The check currently don't analyze code inside macros.


Options
-------

.. option:: CommaPolicy

  Controls whether to add or remove trailing commas.
  Valid values are:

  - `Append`: Add trailing commas where missing.
  - `Remove`: Remove trailing commas where present.

  Example with `CommaPolicy` set to `Append`:

  .. code-block:: c++

    enum Status {
      OK,
      Error     // warning: enum should have a trailing comma
    };

  Example with `CommaPolicy` set to `Remove`:

  .. code-block:: c++

    enum Status {
      OK,
      Error,    // warning: enum should not have a trailing comma
    };

  Default is `Append`.

.. option:: EnumThreshold

  The minimum number of enumerators required in an enum before the check
  will warn. This applies to both `Append` and `Remove` policies.
  Default is `1` (always check enums).

.. option:: InitListThreshold

  The minimum number of elements required in an initializer list before
  the check will warn. This applies to both `Append` and `Remove` policies.
  Default is `3`.
