.. title:: clang-tidy - bugprone-avoid-setjmp-longjmp

bugprone-avoid-setjmp-longjmp
=============================

Flags all call expressions involving ``setjmp()`` and ``longjmp()``.

This check corresponds to the CERT C++ Coding Standard rule
`ERR52-CPP. Do not use setjmp() or longjmp()
<https://www.securecoding.cert.org/confluence/pages/viewpage.action?pageId=88046492>`_.
