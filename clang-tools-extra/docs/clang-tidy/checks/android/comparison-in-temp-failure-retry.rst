.. title:: clang-tidy - android-comparison-in-temp-failure-retry

android-comparison-in-temp-failure-retry
========================================

Diagnoses comparisons that appear to be incorrectly placed in the argument to
the ``TEMP_FAILURE_RETRY`` macro. Having such a use is incorrect in the vast
majority of cases, and will often silently defeat the purpose of the
``TEMP_FAILURE_RETRY`` macro.

For context, ``TEMP_FAILURE_RETRY`` is `a convenience macro
<https://www.gnu.org/software/libc/manual/html_node/Interrupted-Primitives.html>`_
provided by both glibc and Bionic. Its purpose is to repeatedly run a syscall
until it either succeeds, or fails for reasons other than being interrupted.

Example buggy usage looks like:

.. code-block:: c

  char cs[1];
  while (TEMP_FAILURE_RETRY(read(STDIN_FILENO, cs, sizeof(cs)) != 0)) {
    // Do something with cs.
  }

Because TEMP_FAILURE_RETRY will check for whether the result *of the comparison*
is ``-1``, and retry if so.

If you encounter this, the fix is simple: lift the comparison out of the
``TEMP_FAILURE_RETRY`` argument, like so:

.. code-block:: c

  char cs[1];
  while (TEMP_FAILURE_RETRY(read(STDIN_FILENO, cs, sizeof(cs))) != 0) {
    // Do something with cs.
  }

Options
-------

.. option:: RetryMacros

   A comma-separated list of the names of retry macros to be checked.
