.. title:: clang-tidy - bugprone-inc-dec-in-conditions

bugprone-inc-dec-in-conditions
==============================

Detects when a variable is both incremented/decremented and referenced inside a
complex condition and suggests moving them outside to avoid ambiguity in the
variable's value.

When a variable is modified and also used in a complex condition, it can lead to
unexpected behavior. The side-effect of changing the variable's value within the
condition can make the code difficult to reason about. Additionally, the
developer's intended timing for the modification of the variable may not be
clear, leading to misunderstandings and errors. This can be particularly
problematic when the condition involves logical operators like ``&&`` and
``||``, where the order of evaluation can further complicate the situation.

Consider the following example:

.. code-block:: c++

  int i = 0;
  // ...
  if (i++ < 5 && i > 0) {
    // do something
  }

In this example, the result of the expression may not be what the developer
intended. The original intention of the developer could be to increment ``i``
after the entire condition is evaluated, but in reality, i will be incremented
before ``i > 0`` is executed. This can lead to unexpected behavior and bugs in
the code. To fix this issue, the developer should separate the increment
operation from the condition and perform it separately. For example, they can
increment ``i`` in a separate statement before or after the condition is
evaluated. This ensures that the value of ``i`` is predictable and consistent
throughout the code.

.. code-block:: c++

  int i = 0;
  // ...
  i++;
  if (i <= 5 && i > 0) {
    // do something
  }

Another common issue occurs when multiple increments or decrements are performed
on the same variable inside a complex condition. For example:

.. code-block:: c++

  int i = 4;
  // ...
  if (i++ < 5 || --i > 2) {
    // do something
  }

There is a potential issue with this code due to the order of evaluation in C++.
The ``||`` operator used in the condition statement guarantees that if the first
operand evaluates to ``true``, the second operand will not be evaluated. This
means that if ``i`` were initially ``4``, the first operand ``i < 5`` would
evaluate to ``true`` and the second operand ``i > 2`` would not be evaluated.
As a result, the decrement operation ``--i`` would not be executed and ``i``
would hold value ``5``, which may not be the intended behavior for the developer.

To avoid this potential issue, the both increment and decrement operation on
``i`` should be moved outside the condition statement.
