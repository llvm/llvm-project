.. title:: clang-tidy - llvm-formatv-string

llvm-formatv-string
===================

Validates ``llvm::formatv`` format strings against the arguments provided,
similar to how the compiler validates ``printf`` format strings.

This check diagnoses the following issues:

- The number of replacement indices in the format string does not match the
  number of arguments provided.
- A format string does not use one of the given arguments.
- Mixing of automatic and explicit indices (e.g. ``{} {1}``).

.. code-block:: c++

  // warning: formatv() format string requires 2 arguments, but 1 argument was provided
  llvm::formatv("{0} {1}", x);

  // warning: formatv() format string mixes automatic and explicit indices
  llvm::formatv("{} {1}", x, y);

  // warning: formatv() argument unused in format string
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

  For example, to check `mylib::log(Level, const char *Fmt, ...)` set this
  option to `mylib::log:1`. The value `1` indicates the format string is found
  in the second parameter.

  Default is the empty string.
