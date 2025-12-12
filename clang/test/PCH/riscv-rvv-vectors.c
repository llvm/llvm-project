// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +v -emit-pch -o %t %s
// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +v -include-pch %t \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER
#include <riscv_vector.h>
#else
vuint64m4_t v_add(vuint64m4_t a, vuint64m4_t b, size_t vl) {
    return __riscv_vadd_vv_u64m4(a, b, vl);
}
#endif
