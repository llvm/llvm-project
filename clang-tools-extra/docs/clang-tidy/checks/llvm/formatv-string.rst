.. title:: clang-tidy - llvm-formatv-string

llvm-formatv-string
===================

Validates ``llvm::formatv`` format strings against the arguments provided,
similar to how the compiler validates ``printf`` format strings.

This check diagnoses the following issues:

- The number of replacement indices in the format string does not match the
  number of arguments provided.
- A format string does not use an argument at a given index.
- Automatic and explicit indices are mixed (e.g. ``{} {1}``).

.. code-block:: c++

  // Warning: requires 2 arguments, but 1 provided.
  llvm::formatv("{0} {1}", x);

  // Warning: mixes automatic and explicit indices.
  llvm::formatv("{} {1}", x, y);

  // Warning: format string does not use argument at index 1.
  llvm::formatv("{0} {2}", x, y, z);

  // OK.
  llvm::formatv("{0} {1}", x, y);
  llvm::formatv("{} {}", x, y);
  llvm::formatv("{0} {0}", x);

The check only operates on calls where the format string is a string literal.
Dynamic format strings are not diagnosed.

Options
-------

.. option:: AdditionalFunctions

  A semicolon-separated list of additional functions to check, beyond
  ``llvm::formatv``. Each entry has the form `name:index`, where `name` is the
  fully qualified function name and `index` is the zero-based parameter
  position of the format string.

  For example, to check ``mylib::log(Level, const char *Fmt, ...)`` where
  the format string is the second parameter (index 1):

  .. code-block:: yaml

    CheckOptions:
      llvm-formatv-string.AdditionalFunctions: "mylib::log:1"

  Multiple entries are separated by semicolons:

  .. code-block:: yaml

    CheckOptions:
      llvm-formatv-string.AdditionalFunctions: "mylib::log:1;lldb_private::Log::Format:2"

  Default: empty.
