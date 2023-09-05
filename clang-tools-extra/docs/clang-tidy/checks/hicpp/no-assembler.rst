.. title:: clang-tidy - hicpp-no-assembler

hicpp-no-assembler
===================

Check for assembler statements. No fix is offered.

Inline assembler is forbidden by the `High Integrity C++ Coding Standard
<https://www.perforce.com/resources/qac/high-integrity-cpp-coding-standard/declarations>`_
as it restricts the portability of code.
