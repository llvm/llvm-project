// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -Wno-non-pod-varargs -fclangir -emit-llvm %s -o - 2>&1 | FileCheck %s

// A non-trivially-copyable class is passed as a reference to caller-owned
// storage, which the fetch cannot yet read.  Reaching it needs
// -Wno-non-pod-varargs.
struct NonTrivial {
  int a;
  NonTrivial(const NonTrivial &);
  ~NonTrivial();
};

NonTrivial take_non_pod(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  NonTrivial res = __builtin_va_arg(args, NonTrivial);
  __builtin_va_end(args);
  return res;
}

// CHECK: error: 'cir.va_arg' op va_arg of a non-trivially-copyable type not yet implemented in CallConvLowering
