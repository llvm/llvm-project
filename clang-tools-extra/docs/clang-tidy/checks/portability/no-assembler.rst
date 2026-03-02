.. title:: clang-tidy - portability-no-assembler

portability-no-assembler
========================

Checks for assembler statements. Use of inline assembly should be avoided
since it restricts the portability of the code.

.. code-block:: c++

   asm("mov al, 2");  // warning: do not use assembler statements

`hicpp-no-assembler` is an alias for this check that enforces rule 7.5.1 of
the High Integrity C++ Coding Standard.