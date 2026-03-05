// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fno-clangir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

#include <stdarg.h>

double f1(int cond, int n, ...) {
  va_list valist;
  va_start(valist, n);
  double res = cond ? va_arg(valist, double) : 0;
  va_end(valist);
  return res;
}

// Fine enough to check it passes the verifying.
// CIR: cir.ternary

int unconditional_evaluation(_Bool cond) {
  return cond ? 123 : 456;
  // CIR: %[[TRUE_CONST:.+]] = cir.const #cir.int<123>
  // CIR: %[[FALSE_CONST:.+]] = cir.const #cir.int<456>
  // CIR: cir.select if {{.+}} then %[[TRUE_CONST]] else %[[FALSE_CONST]] : (!cir.bool, !s32i, !s32i) -> !s32i
}
