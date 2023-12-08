.. title:: clang-tidy - linuxkernel-must-use-errs

linuxkernel-must-use-errs
=========================

Checks Linux kernel code to see if it uses the results from the functions in
``linux/err.h``. Also checks to see if code uses the results from functions that
directly return a value from one of these error functions.

This is important in the Linux kernel because ``ERR_PTR``, ``PTR_ERR``,
``IS_ERR``, ``IS_ERR_OR_NULL``, ``ERR_CAST``, and ``PTR_ERR_OR_ZERO`` return
values must be checked, since positive pointers and negative error codes are
being used in the same context. These functions are marked with
``__attribute__((warn_unused_result))``, but some kernel versions do not have
this warning enabled for clang.

Examples:

.. code-block:: c

  /* Trivial unused call to an ERR function */
  PTR_ERR_OR_ZERO(some_function_call());

  /* A function that returns ERR_PTR. */
  void *fn() { ERR_PTR(-EINVAL); }

  /* An invalid use of fn. */
  fn();
