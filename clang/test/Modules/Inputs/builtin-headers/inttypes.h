#ifndef INTTYPES_H
#define INTTYPES_H

// This creates an include cycle when inttypes.h and stdint.h
// are both part of the cstd module. This include will resolve
// to the C++ stdint.h, which will #include_next eventually to
// the stdint.h in this directory, and thus create the cycle
// cstd (inttypes.h) -> cpp_stdint (stdint.h) -> cstd (stdint.h).
// This cycle is worked around by cstd using [no_undeclared_includes].
#include <stdint.h>

#endif
