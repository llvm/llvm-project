// RUN: %check_clang_tidy -std=c++20-or-later \
// RUN:   -check-header %S/Inputs/use-bit-cast/header.h \
// RUN:   %s modernize-use-bit-cast %t -- \
// RUN:   -- -I%S/Inputs/use-bit-cast

void *memcpy(void *To, const void *From, unsigned long long Size);

namespace std {
using ::memcpy;
}

#include "header.h"
#include "header.h"
