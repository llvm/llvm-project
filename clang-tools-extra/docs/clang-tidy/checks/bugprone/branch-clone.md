.. title:: clang-tidy - bugprone-branch-clone

bugprone-branch-clone
=====================

Checks for repeated branches in ``if/else if/else`` chains, consecutive
repeated branches in ``switch`` statements and identical true and false
branches in conditional operators.

.. code-block:: c++

    if (test_value(x)) {
      y++;
      do_something(x, y);
    } else {
      y++;
      do_something(x, y);
    }

In this simple example (which could arise e.g. as a copy-paste error) the
``then`` and ``else`` branches are identical and the code is equivalent the
following shorter and cleaner code:

.. code-block:: c++

    test_value(x); // can be omitted unless it has side effects
    y++;
    do_something(x, y);


If this is the intended behavior, then there is no reason to use a conditional
statement; otherwise the issue can be solved by fixing the branch that is
handled incorrectly.

The check detects repeated branches in longer ``if/else if/else`` chains
where it would be even harder to notice the problem.

The check also detects repeated inner and outer ``if`` statements that may
be a result of a copy-paste error. This check cannot currently detect
identical inner and outer ``if`` statements if code is between the ``if``
conditions. An example is as follows.

.. code-block:: c++

    void test_warn_inner_if_1(int x) {
      if (x == 1) {    // warns, if with identical inner if
        if (x == 1)    // inner if is here
          ;
      if (x == 1) {    // does not warn, cannot detect
        int y = x;
        if (x == 1)
          ;
      }
    }


In ``switch`` statements the check only reports repeated branches when they are
consecutive, because it is relatively common that the ``case:`` labels have
some natural ordering and rearranging them would decrease the readability of
the code. For example:

.. code-block:: c++

    switch (ch) {
    case 'a':
      return 10;
    case 'A':
      return 10;
    case 'b':
      return 11;
    case 'B':
      return 11;
    default:
      return 10;
    }

Here the check reports that the ``'a'`` and ``'A'`` branches are identical
(and that the ``'b'`` and ``'B'`` branches are also identical), but does not
report that the ``default:`` branch is also identical to the first two branches.
If this is indeed the correct behavior, then it could be implemented as:

.. code-block:: c++

    switch (ch) {
    case 'a':
    case 'A':
      return 10;
    case 'b':
    case 'B':
      return 11;
    default:
      return 10;
    }

Here the check does not warn for the repeated ``return 10;``, which is good if
we want to preserve that ``'a'`` is before ``'b'`` and ``default:`` is the last
branch.

Switch cases marked with the ``[[fallthrough]]`` attribute are ignored.

Finally, the check also examines conditional operators and reports code like:

.. code-block:: c++

    return test_value(x) ? x : x;

Unlike if statements, the check does not detect chains of conditional
operators.

Note: This check also reports situations where branches become identical only
after preprocessing.
