// RUN: %clang_cc1 -triple arm-unknown-linux-gnueabi -emit-llvm %s -o - | FileCheck -check-prefix=LIBCALL %s
// RUN: %clang_cc1 -triple armv8-eabi -emit-llvm %s -o - | FileCheck -check-prefix=NATIVE %s
// PR45476

// This test used to get into an infinite loop,
// which, in turn, caused clang to never finish execution.

struct s3 {
  char a, b, c;
};

_Atomic struct s3 a;

extern "C" void foo() {
  // LIBCALL-LABEL: @foo
  // LIBCALL: call void @__atomic_store
  // NATIVE-LABEL: @foo
  // NATIVE: store atomic i32 {{.*}} seq_cst, align 4

  a = s3{1, 2, 3};
}
