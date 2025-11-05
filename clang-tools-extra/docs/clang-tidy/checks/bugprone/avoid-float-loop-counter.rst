.. title:: clang-tidy - bugprone-avoid-float-loop-counter

bugprone-avoid-float-loop-counter
=================================

Flags ``for`` loops where the induction expression has a floating-point type.

References
----------

This check corresponds to the CERT C Coding Standard rule
`FLP30-C. Do not use floating-point variables as loop counters
<https://www.securecoding.cert.org/confluence/display/c/FLP30-C.+Do+not+use+floating-point+variables+as+loop+counters>`_.
