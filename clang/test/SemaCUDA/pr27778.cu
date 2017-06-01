// RUN: %clang_cc1 -fsyntax-only %s

#include "Inputs/cuda.h"

const int constint = 512;
__launch_bounds__(constint) void TestConstInt(void) {}
