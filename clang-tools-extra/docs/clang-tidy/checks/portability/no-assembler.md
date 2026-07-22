.. title:: clang-tidy - portability-no-assembler

portability-no-assembler
========================

Checks for assembler statements. Use of inline assembly should be avoided
since it ties to a specific CPU architecture and syntax making code that
uses it non-portable across platforms.

.. code-block:: c++

   asm("mov al, 2");  // warning: do not use assembler statements
