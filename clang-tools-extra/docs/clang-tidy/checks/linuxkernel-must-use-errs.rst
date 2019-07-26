.. title:: clang-tidy - linuxkernel-must-use-errs

linuxkernel-must-use-errs
=========================

Checks for cases where the kernel error functions ``ERR_PTR``,
``PTR_ERR``, ``IS_ERR``, ``IS_ERR_OR_NULL``, ``ERR_CAST``, and
``PTR_ERR_OR_ZERO`` are called but the results are not used. These
functions are marked with ``__attribute__((warn_unused_result))``, but
the compiler warning for this attribute is not always enabled.

This also checks for unused values returned by functions that return
``ERR_PTR``.

Examples:

.. code-block:: c

  /* Trivial unused call to an ERR function */
  PTR_ERR_OR_ZERO(some_function_call());

  /* A function that returns ERR_PTR. */
  void *fn() { ERR_PTR(-EINVAL); }

  /* An invalid use of fn. */
  fn();
