// RUN: %clang_cc1 -triple riscv64 -fsyntax-only \
// RUN:   -target-feature +m -target-feature +a -target-feature +f \
// RUN:   -target-feature +d -target-feature +experimental-v %s
// expected-no-diagnostics

#include <riscv_vector.h>
