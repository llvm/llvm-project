.. title:: clang-tidy - portability-deprecated-posix-functions

portability-deprecated-posix-functions
======================================

Finds uses of deprecated or obsolete POSIX functions and suggests modern
replacements.

The following functions are checked:

- ``bcmp``, suggested replacement: ``memcmp``
- ``bcopy``, suggested replacement: ``memmove``
- ``bzero``, suggested replacement: ``memset``
- ``getpw``, suggested replacement: ``getpwuid``
- ``vfork``, suggested replacement: ``posix_spawn``
