// RUN: %clang -O2 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=INLINE
// RUN: %clang -O2 -S -emit-llvm %s -fno-inline-functions-called-once -o - | FileCheck %s --check-prefix=NOINLINE

// INLINE-LABEL: define{{.*}}@main
// INLINE-NOT: call{{.*}}@_ZL12bad_functionv
// INLINE: ret i32 0

// NOINLINE-LABEL: define{{.*}}@_ZL4testv
// NOINLINE: call{{.*}}@_ZL12bad_functionv
// NOINLINE: ret void

// NOINLINE: define internal{{.*}}@_ZL12bad_functionv

volatile int G;

static void bad_function(void) {
  G++;
}

static void test(void) {
  bad_function();
}

int main(void) {
  test();
  return 0;
}
