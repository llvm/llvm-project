.. title:: clang-tidy - bugprone-chained-comparison

bugprone-chained-comparison
===========================

Check detects chained comparison operators that can lead to unintended
behavior or logical errors.

Chained comparisons are expressions that use multiple comparison operators
to compare three or more values. For example, the expression ``a < b < c``
compares the values of ``a``, ``b``, and ``c``. However, this expression does
not evaluate as ``(a < b) && (b < c)``, which is probably what the developer
intended. Instead, it evaluates as ``(a < b) < c``, which may produce
unintended results, especially when the types of ``a``, ``b``, and ``c`` are
different.

To avoid such errors, the check will issue a warning when a chained
comparison operator is detected, suggesting to use parentheses to specify
the order of evaluation or to use a logical operator to separate comparison
expressions.

Consider the following examples:

.. code-block:: c++

    int a = 2, b = 6, c = 4;
    if (a < b < c) {
        // This block will be executed
    }


In this example, the developer intended to check if ``a`` is less than ``b``
and ``b`` is less than ``c``. However, the expression ``a < b < c`` is
equivalent to ``(a < b) < c``. Since ``a < b`` is ``true``, the expression
``(a < b) < c`` is evaluated as ``1 < c``, which is equivalent to ``true < c``
and is invalid in this case as ``b < c`` is ``false``.

Even that above issue could be detected as comparison of ``int`` to ``bool``,
there is more dangerous example:

.. code-block:: c++

    bool a = false, b = false, c = true;
    if (a == b == c) {
        // This block will be executed
    }

In this example, the developer intended to check if ``a``, ``b``, and ``c`` are
all equal. However, the expression ``a == b == c`` is evaluated as
``(a == b) == c``. Since ``a == b`` is true, the expression ``(a == b) == c``
is evaluated as ``true == c``, which is equivalent to ``true == true``.
This comparison yields ``true``, even though ``a`` and ``b`` are ``false``, and
are not equal to ``c``.

To avoid this issue, the developer can use a logical operator to separate the
comparison expressions, like this:

.. code-block:: c++

    if (a == b && b == c) {
        // This block will not be executed
    }


Alternatively, use of parentheses in the comparison expressions can make the
developer's intention more explicit and help avoid misunderstanding.

.. code-block:: c++

    if ((a == b) == c) {
        // This block will be executed
    }

