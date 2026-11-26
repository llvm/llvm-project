.. title:: clang-tidy - llvm-formatv-string

llvm-formatv-string
===================

Validates ``llvm::formatv`` format strings against the provided arguments,
diagnosing mismatched argument counts, unused arguments, and mixed index
styles.

This check diagnoses the following issues:

- The number of replacement indices in the format string does not match the
  number of arguments provided.
- A format string does not use one of the given arguments.
- Mixing of automatic and explicit indices (e.g. ``{} {1}``).

.. code-block:: c++

  // warning: format string requires 2 arguments, but 1 argument was provided
  llvm::formatv("{0} {1}", x);

  // warning: format string mixes automatic and explicit indices
  llvm::formatv("{} {1}", x, y);

  // warning: argument unused in format string
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

  A semicolon-separated list of additional fully qualified function names to
  check, beyond ``llvm::formatv`` and ``llvm::createStringErrorV``. Each
  function must be a variadic template whose last parameter is a parameter
  pack. The format string is assumed to be the parameter immediately preceding
  the pack.

  For example, to check ``::mylib::log(Level, const char *Fmt, Ts&&...)`` set
  this option to `::mylib::log`.

  Default is the empty string.
