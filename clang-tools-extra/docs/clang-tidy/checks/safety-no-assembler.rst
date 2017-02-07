.. title:: clang-tidy - safety-no-assembler

safety-no-assembler
===================

Check for assembler statements. No fix is offered.

Inline assembler is forbidden by safety-critical C++ standards like `High
Intergrity C++ <http://www.codingstandard.com>`_ as it restricts the
portability of code.
