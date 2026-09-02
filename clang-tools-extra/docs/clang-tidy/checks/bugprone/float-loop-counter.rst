.. title:: clang-tidy - bugprone-float-loop-counter

bugprone-float-loop-counter
===========================

Flags ``for`` loops where the induction expression has a floating-point type.

References
----------

This check corresponds to the CERT C Coding Standard rule
`FLP30-C. Do not use floating-point variables as loop counters
<https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/rules/floating-point-flp/flp30-c/>`_.
