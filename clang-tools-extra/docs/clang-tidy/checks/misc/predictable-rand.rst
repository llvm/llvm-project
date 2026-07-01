.. title:: clang-tidy - misc-predictable-rand

misc-predictable-rand
=====================

Warns for the usage of ``std::rand()``. Pseudorandom number generators use
mathematical algorithms to produce a sequence of numbers with good
statistical properties, but the numbers produced are not genuinely random.
The ``std::rand()`` function takes a seed (number), runs a mathematical
operation on it and returns the result. By manipulating the seed the result
can be predictable.

References
----------

This check corresponds to the CERT C Coding Standard rules
`MSC30-C. Do not use the rand() function for generating pseudorandom numbers
<https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/rules/miscellaneous-msc/msc30-c/>`_.
`MSC50-CPP. Do not use std::rand() for generating pseudorandom numbers
<https://cmu-sei.github.io/secure-coding-standards/sei-cert-cpp-coding-standard/rules/miscellaneous-msc/msc50-cpp/>`_.
