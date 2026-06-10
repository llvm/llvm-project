// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

struct HasFloat {
  float f;
  int i;
};

void take_float(int n, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, n);
  struct HasFloat hf = __builtin_va_arg(args, struct HasFloat);
  __builtin_va_end(args);
}

// CHECK: error: ClangIR code gen Not Yet Implemented: va_arg of an aggregate type with non-integer members
