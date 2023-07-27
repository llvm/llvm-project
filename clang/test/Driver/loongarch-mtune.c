// RUN: %clang --target=loongarch64 -mtune=loongarch64 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1ARG -DCPU=loongarch64
// RUN: %clang --target=loongarch64 -mtune=loongarch64 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IRATTR -DCPU=loongarch64
//
// RUN: %clang --target=loongarch64 -mtune=la464 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1ARG -DCPU=la464
// RUN: %clang --target=loongarch64 -mtune=la464 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IRATTR -DCPU=la464

// CC1ARG: "-tune-cpu" "[[CPU]]"
// IRATTR: "tune-cpu"="[[CPU]]"

int foo(void) {
  return 3;
}
