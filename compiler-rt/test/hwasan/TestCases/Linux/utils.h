#pragma once

#include <stdint.h>

#if defined(__x86_64__)
#define UNTAG(x) (x)
#else
#define UNTAG(x) (typeof((x) + 0))(((uintptr_t)(x)) & 0xffffffffffffff)
#endif
