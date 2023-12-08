#ifndef J_H
#define J_H

#define STR(x) #x
#define HDR(x) STR(x.h)

#include ALLOWED_INC
#include HDR(a)

const int j = a * a + b;

// expected-no-diagnostics

#endif
