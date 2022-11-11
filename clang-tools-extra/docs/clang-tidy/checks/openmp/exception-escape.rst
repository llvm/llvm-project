.. title:: clang-tidy - openmp-exception-escape

openmp-exception-escape
=======================

Analyzes OpenMP Structured Blocks and checks that no exception escapes
out of the Structured Block it was thrown in.

As per the OpenMP specification, a structured block is an executable statement,
possibly compound, with a single entry at the top and a single exit at the
bottom. Which means, ``throw`` may not be used to 'exit' out of the
structured block. If an exception is not caught in the same structured block
it was thrown in, the behavior is undefined.

FIXME: this check does not model SEH, ``setjmp``/``longjmp``.

WARNING! This check may be expensive on large source files.

Options
-------

.. option:: IgnoredExceptions

   Comma-separated list containing type names which are not counted as thrown
   exceptions in the check. Default value is an empty string.
