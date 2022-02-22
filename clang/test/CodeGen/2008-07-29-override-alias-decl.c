// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s

int x(void) { return 1; }

// CHECK:  ret i32 1


int f(void) __attribute__((weak, alias("x")));

/* Test that we link to the alias correctly instead of making a new
   forward definition. */
int f(void);
int h(void) {
  return f();
}

// CHECK:  [[call:%.*]] = call i32 @f()
// CHECK:  ret i32 [[call]]

