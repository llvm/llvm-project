.. title:: clang-tidy - cert-flp30-c

cert-flp30-c
============

This check flags ``for`` loops where the induction expression has a
floating-point type.

This check corresponds to the CERT C Coding Standard rule
`FLP30-C. Do not use floating-point variables as loop counters
<https://www.securecoding.cert.org/confluence/display/c/FLP30-C.+Do+not+use+floating-point+variables+as+loop+counters>`_.
