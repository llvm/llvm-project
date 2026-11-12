// RUN: %clang_cc1 -E %s

#define test
#include "pr2086.h"
#define test
#include "pr2086.h"

#ifdef test
#error
#endif

