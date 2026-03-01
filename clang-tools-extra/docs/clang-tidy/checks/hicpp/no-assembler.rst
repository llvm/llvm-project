.. title:: clang-tidy - hicpp-no-assembler
.. meta::
   :http-equiv=refresh: 0;URL=../portability/no-assembler.html
   
hicpp-no-assembler
==================

Checks for assembler statements. Use of inline assembly should be avoided since
it restricts the portability of the code.

This enforces `rule 7.5.1 <https://www.perforce.com/resources/qac/high-integrity-cpp-coding-rules>`_
of the High Integrity C++ Coding Standard.
