.. title:: clang-tidy - bugprone-infinite-loop

bugprone-infinite-loop
======================

Finds obvious infinite loops (loops where the condition variable is not changed
at all).

Finding infinite loops is well-known to be impossible (halting problem).
However, it is possible to detect some obvious infinite loops, for example, if
the loop condition is not changed. This check detects such loops. A loop is
considered infinite if it does not have any loop exit statement (``break``,
``continue``, ``goto``, ``return``, ``throw`` or a call to a function called as
``[[noreturn]]``) and all of the following conditions hold for every variable in
the condition:

- It is a local variable.
- It has no reference or pointer aliases.
- It is not a structure or class member.

Furthermore, the condition must not contain a function call to consider the loop
infinite since functions may return different values for different calls.

For example, the following loop is considered infinite `i` is not changed in
the body:

.. code-block:: c++

  int i = 0, j = 0;
  while (i < 10) {
    ++j;
  }
