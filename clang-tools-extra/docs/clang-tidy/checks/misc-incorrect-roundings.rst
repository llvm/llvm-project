misc-incorrect-roundings
========================

Checks the usage of patterns known to produce incorrect rounding.
Programmers often use::

   (int)(double_expression + 0.5)

to round the double expression to an integer. The problem with this:

1. It is unnecessarily slow.
2. It is incorrect. The number 0.499999975 (smallest representable float
   number below 0.5) rounds to 1.0. Even worse behavior for negative
   numbers where both -0.5f and -1.4f both round to 0.0.
