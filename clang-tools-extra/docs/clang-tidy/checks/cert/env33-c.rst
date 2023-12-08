.. title:: clang-tidy - cert-env33-c

cert-env33-c
============

This check flags calls to ``system()``, ``popen()``, and ``_popen()``, which
execute a command processor. It does not flag calls to ``system()`` with a null
pointer argument, as such a call checks for the presence of a command processor
but does not actually attempt to execute a command.

This check corresponds to the CERT C Coding Standard rule
`ENV33-C. Do not call system()
<https://www.securecoding.cert.org/confluence/pages/viewpage.action?pageId=2130132>`_.
