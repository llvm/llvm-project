This directory contains partial re-implementations of some C++ standard library
utilities. They are for use with internal LLVM libc code and tests.

More utilities can be added on an as needed basis. There are certain rules to
be followed for future changes and additions:

1. Only two kind of headers can be included: Other headers from this directory,
and free standing C headers.
2. Free standing C headers are to be included as C headers and not as C++
headers. That is, use `#include <stddef.h>` and not `#include <cstddef>`.
3. The utilities should be defined in the namespace `LIBC_NAMESPACE::cpp`. The
higher level namespace should have a `__` prefix to avoid symbol name pollution
when the utilities are used in implementation of public functions.
