.. title:: clang-tidy - portability-no-assembler

portability-no-assembler
========================

Checks for assembler statements. Use of inline assembly should be avoided
since it restricts the portability of the code.

.. code-block:: c++

   asm("mov al, 2");  // warning: do not use assembler statements
