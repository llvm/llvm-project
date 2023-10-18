.. title:: clang-tidy - bugprone-casting-through-void

bugprone-casting-through-void
=============================

A check detects usage of cast pointer to the other pointer throght cast to
``void *`` in C/C++ code.

Use of these casts can violate type safety and cause the program to access a
variable that is actually of type ``X`` to be accessed as if it were of an
unrelated type ``Z``.
