.. title:: clang-tidy - readability-use-rethrow

readability-use-rethrow
=======================

Detects cases where a caught exception is explicitly re-thrown as a new copy,
and suggests using a bare ``throw;`` instead.

Throwing a caught exception by its variable name (e.g., ``throw e;``) instead
of using a bare ``throw;`` creates a new copy of the exception. This can lead
to object slicing if the exception was derived from the caught type, and it
alters the original stack trace of the exception.

This check only flags exceptions caught by reference (``const`` or
non-``const``) to avoid false positives where an exception is caught by value,
modified, and then thrown again.

Example:

.. code-block:: c++

  try {
    // ...
  } catch (const std::exception &e) {
    log(e.what());
    throw e; // Warning: throwing a copy of the caught exception
  }

Is transformed to:

.. code-block:: c++

  try {
    // ...
  } catch (const std::exception &e) {
    log(e.what());
    throw;
  }
