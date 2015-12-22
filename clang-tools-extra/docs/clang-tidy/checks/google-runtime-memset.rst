.. title:: clang-tidy - google-runtime-memset

google-runtime-memset
=====================


Finds calls to memset with a literal zero in the length argument.

This is most likely unintended and the length and value arguments are
swapped.

Corresponding cpplint.py check name: 'runtime/memset'.
