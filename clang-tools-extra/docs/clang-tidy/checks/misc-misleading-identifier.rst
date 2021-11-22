.. title:: clang-tidy - misc-misleading-identifier

misc-misleading-identifier
==========================

Finds identifiers that contain Unicode characters with right-to-left direction,
which can be confusing as they may change the understanding of a whole statement
line, as described in `Trojan Source <https://trojansource.codes>`_.

An example of such misleading code follows:

.. code-block:: text

  #include <stdio.h>

  short int א = (short int)0;
  short int ג = (short int)12345;

  int main() {
    int א = ג; // a local variable, set to zero?
    printf("ג is %d\n", ג);
    printf("א is %d\n", א);
  }
