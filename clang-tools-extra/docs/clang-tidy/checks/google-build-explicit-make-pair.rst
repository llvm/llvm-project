.. title:: clang-tidy - google-build-explicit-make-pair

google-build-explicit-make-pair
===============================

Check that ``make_pair``'s template arguments are deduced.

G++ 4.6 in C++11 mode fails badly if ``make_pair``'s template arguments are
specified explicitly, and such use isn't intended in any case.

Corresponding cpplint.py check name: 'build/explicit_make_pair'.
