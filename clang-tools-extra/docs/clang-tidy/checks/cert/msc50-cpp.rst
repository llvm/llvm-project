.. title:: clang-tidy - cert-msc50-cpp

cert-msc50-cpp
==============

Pseudorandom number generators use mathematical algorithms to produce a sequence
of numbers with good statistical properties, but the numbers produced are not
genuinely random. The ``std::rand()`` function takes a seed (number), runs a
mathematical operation on it and returns the result. By manipulating the seed
the result can be predictable. This check warns for the usage of
``std::rand()``.
