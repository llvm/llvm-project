// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

struct Big {
  __int128 x;
};

void take_overaligned(int n, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, n);
  struct Big b = __builtin_va_arg(args, struct Big);
  __builtin_va_end(args);
}

// CHECK: error: ClangIR code gen Not Yet Implemented: va_arg of an over-aligned aggregate type
